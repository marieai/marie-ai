import asyncio
import json
import select
from typing import Any, Callable, Dict, Optional

import psycopg2
import psycopg2.extensions

from marie.excepts import RuntimeFailToStart
from marie.logging_core.logger import MarieLogger


class NotificationService:
    """
    Service for handling PostgreSQL LISTEN/NOTIFY notifications.

    This service manages a dedicated PostgreSQL connection for listening to
    database events and routing them to registered handlers. It runs in a
    separate async task and does not block the main event loop.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification service.

        :param config: Database configuration with connection parameters
        """
        self.logger = MarieLogger(NotificationService.__name__)
        self.config = config
        self.running = False

        # Dedicated connection for LISTEN operations (cannot use pool)
        self._listen_connection: Optional[psycopg2.extensions.connection] = None
        self._listener_task: Optional[asyncio.Task] = None

        # Map of channel names to handler callbacks
        self._handlers: Dict[str, Callable] = {}

        # Channels to listen on
        self._channels: set[str] = set()

    def register_handler(self, channel: str, handler: Callable) -> None:
        """
        Register a handler for a specific notification channel.

        :param channel: PostgreSQL notification channel name
        :param handler: Async callback function to handle notifications.
                       Should accept a dict payload parameter.
        """
        self._handlers[channel] = handler
        self._channels.add(channel)
        self.logger.info(f"Registered handler for channel: {channel}")

    def unregister_handler(self, channel: str) -> None:
        """
        Unregister a handler for a channel.

        :param channel: Channel name to unregister
        """
        if channel in self._handlers:
            del self._handlers[channel]
            self._channels.discard(channel)
            self.logger.info(f"Unregistered handler for channel: {channel}")

    def _setup_connection(self) -> None:
        """
        Set up dedicated PostgreSQL connection for LISTEN operations.
        This runs in a thread pool executor to avoid blocking the event loop.
        """
        try:
            self.logger.info("Setting up PostgreSQL LISTEN connection")

            config = self.config
            self._listen_connection = psycopg2.connect(
                user=config["username"],
                password=config["password"],
                database=config["database"],
                host=config["hostname"],
                port=int(config["port"]),
                options='-c timezone=UTC',
                application_name=f"{config.get('application_name', 'marie_scheduler')}_listener",
            )

            # Set isolation level to autocommit for LISTEN/NOTIFY
            self._listen_connection.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
            )

            # Register LISTEN for all configured channels
            cursor = self._listen_connection.cursor()
            for channel in self._channels:
                cursor.execute(f"LISTEN {channel};")
                self.logger.info(f"Listening on channel: {channel}")
            cursor.close()

            self.logger.info("PostgreSQL LISTEN connection established successfully")

        except Exception as e:
            self.logger.error(f"Failed to set up LISTEN connection: {e}")
            raise RuntimeFailToStart(
                f"Failed to set up PostgreSQL LISTEN connection: {e}"
            )

    def _close_connection(self) -> None:
        """
        Close the PostgreSQL LISTEN connection.
        """
        if self._listen_connection and not self._listen_connection.closed:
            try:
                self.logger.info("Closing PostgreSQL LISTEN connection")
                self._listen_connection.close()
            except Exception as e:
                self.logger.warning(f"Error closing LISTEN connection: {e}")
            finally:
                self._listen_connection = None

    async def start(self) -> None:
        """
        Start the notification service and begin listening for notifications.
        """
        if self.running:
            self.logger.warning("NotificationService is already running")
            return

        if not self._channels:
            self.logger.warning(
                "No channels registered. NotificationService will not start."
            )
            return

        self.logger.info("Starting NotificationService")
        self.running = True

        # Start the listener task
        self._listener_task = asyncio.create_task(self._listen_for_notifications())

        # Wait briefly to catch early failures
        await asyncio.sleep(0.5)
        if self._listener_task.done():
            # Re-raise any exception from the listener task
            await self._listener_task

    async def stop(self) -> None:
        """
        Stop the notification service and cleanup resources.
        """
        if not self.running:
            return

        self.logger.info("Stopping NotificationService")
        self.running = False

        # Cancel the listener task
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        # Close the connection (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._close_connection)

        self.logger.info("NotificationService stopped")

    async def _listen_for_notifications(self) -> None:
        """
        Main listening loop for PostgreSQL notifications.

        This method runs in the background and continuously checks for
        notifications from the database. It uses select() in a thread pool
        to avoid blocking the event loop.
        """
        try:
            # Set up the connection in executor (blocking operation)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._setup_connection)

            self.logger.info("Notification listener loop started")

            while self.running:
                # Use select() in executor to wait for notifications without blocking
                # Timeout of 1.0 second allows checking self.running regularly
                ready = await loop.run_in_executor(
                    None, lambda: select.select([self._listen_connection], [], [], 1.0)
                )

                # If select() returned empty, no notification arrived
                if ready == ([], [], []):
                    continue

                # Poll for notifications (also in executor to avoid blocking)
                await loop.run_in_executor(None, self._listen_connection.poll)

                # Process all pending notifications
                while self._listen_connection.notifies:
                    notify = self._listen_connection.notifies.pop(0)

                    try:
                        # Parse JSON payload
                        payload = json.loads(notify.payload)
                        channel = notify.channel

                        self.logger.debug(
                            f"Received notification on channel '{channel}': {payload}"
                        )

                        # Route to registered handler
                        if channel in self._handlers:
                            handler = self._handlers[channel]
                            try:
                                # Call handler (should be async)
                                await handler(payload)
                            except Exception as e:
                                self.logger.error(
                                    f"Error in handler for channel '{channel}': {e}",
                                    exc_info=True,
                                )
                        else:
                            self.logger.warning(
                                f"No handler registered for channel '{channel}'"
                            )

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse notification payload: {e}")
                    except Exception as e:
                        self.logger.error(
                            f"Error processing notification: {e}", exc_info=True
                        )

        except asyncio.CancelledError:
            self.logger.info("Notification listener task cancelled")
            raise

        except Exception as e:
            self.logger.error(
                f"Fatal error in notification listener: {e}", exc_info=True
            )
            # Re-raise to fail the start() call
            raise RuntimeFailToStart(f"Notification listener failed: {e}")

        finally:
            # Ensure connection is closed
            await asyncio.get_event_loop().run_in_executor(None, self._close_connection)

    async def send_notification(self, channel: str, payload: Dict[str, Any]) -> bool:
        """
        Send a notification to a PostgreSQL channel.

        Note: This requires a separate connection and is not typically used
        by the listener. Usually the database triggers send notifications.

        :param channel: Channel name
        :param payload: Notification payload (will be JSON-encoded)
        :return: True if successful, False otherwise
        """
        try:
            # Create temporary connection for sending
            config = self.config
            conn = psycopg2.connect(
                user=config["username"],
                password=config["password"],
                database=config["database"],
                host=config["hostname"],
                port=int(config["port"]),
            )
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            cursor = conn.cursor()
            payload_json = json.dumps(payload)
            cursor.execute(f"SELECT pg_notify(%s, %s)", (channel, payload_json))
            cursor.close()
            conn.close()

            self.logger.debug(f"Sent notification to channel '{channel}': {payload}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False

    def is_running(self) -> bool:
        """
        Check if the notification service is running.

        :return: True if running, False otherwise
        """
        return self.running

    def get_registered_channels(self) -> set[str]:
        """
        Get the set of registered channels.

        :return: Set of channel names
        """
        return self._channels.copy()
