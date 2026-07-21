import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

import psycopg

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
        self._listen_connection: Optional[psycopg.Connection] = None
        self._listener_task: Optional[asyncio.Task] = None

        # Map of channel names to handler callbacks
        self._handlers: Dict[str, Callable] = {}

        # Channels to listen on
        self._channels: set[str] = set()
        self._ready_event = asyncio.Event()
        self._connected = False
        self._ever_connected = False
        self._last_notification_at: Optional[float] = None
        self._reconnect_base_delay = 1.0
        self._reconnect_max_delay = 30.0
        self._select_timeout = 1.0

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
            self._listen_connection = psycopg.connect(
                user=config["username"],
                password=config["password"],
                dbname=config["database"],
                host=config["hostname"],
                port=int(config["port"]),
                options='-c timezone=UTC',
                application_name=f"{config.get('application_name', 'marie_scheduler')}_listener",
                keepalives=1,
                keepalives_idle=60,
                keepalives_interval=10,
                keepalives_count=5,
            )

            self._listen_connection.autocommit = True

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
        self._ready_event.clear()
        self._connected = False
        self._ever_connected = False

        # Start the listener task
        self._listener_task = asyncio.create_task(self._listen_for_notifications())

        ready_task = asyncio.create_task(self._ready_event.wait())
        done, _pending = await asyncio.wait(
            {ready_task, self._listener_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready_task in done:
            return

        ready_task.cancel()
        await self._listener_task

    async def stop(self) -> None:
        """
        Stop the notification service and cleanup resources.
        """
        if not self.running:
            return

        self.logger.info("Stopping NotificationService")
        self.running = False
        self._connected = False
        self._ready_event.clear()

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

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_notification_at(self) -> Optional[float]:
        return self._last_notification_at

    async def _listen_for_notifications(self) -> None:
        """
        Main listening loop for PostgreSQL notifications.

        This method runs in the background and continuously checks for
        notifications from the database. It uses select() in a thread pool
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        reconnect_delay = self._reconnect_base_delay

        try:
            while self.running:
                try:
                    await loop.run_in_executor(None, self._setup_connection)
                    self._connected = True
                    self._ever_connected = True
                    reconnect_delay = self._reconnect_base_delay
                    self._ready_event.set()
                    self.logger.info("Notification listener loop started")

                    while self.running:
                        if (
                            self._listen_connection is None
                            or self._listen_connection.closed
                        ):
                            raise RuntimeError("LISTEN connection is closed")

                        notify = await loop.run_in_executor(
                            None, self._next_notification
                        )
                        if notify is None:
                            continue
                        self._last_notification_at = time.monotonic()

                        try:
                            payload = json.loads(notify.payload)
                            channel = notify.channel
                            self.logger.debug(
                                f"Received notification on channel '{channel}': {payload}"
                            )
                            if channel in self._handlers:
                                handler = self._handlers[channel]
                                try:
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
                            self.logger.error(
                                f"Failed to parse notification payload: {e}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Error processing notification: {e}", exc_info=True
                            )
                except asyncio.CancelledError:
                    self.logger.info("Notification listener task cancelled")
                    raise
                except Exception as e:
                    self._connected = False
                    await loop.run_in_executor(None, self._close_connection)
                    if not self._ever_connected:
                        self.logger.error(
                            f"Fatal error in notification listener: {e}",
                            exc_info=True,
                        )
                        raise RuntimeFailToStart(
                            f"Notification listener failed: {e}"
                        ) from e

                    if not self.running:
                        break

                    self.logger.warning(
                        "Notification listener lost connection: "
                        f"{e}. Reconnecting in {reconnect_delay:.1f}s"
                    )
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2, self._reconnect_max_delay
                    )
        finally:
            self._connected = False
            await asyncio.get_event_loop().run_in_executor(None, self._close_connection)

    def _next_notification(self) -> Optional[psycopg.Notify]:
        if self._listen_connection is None or self._listen_connection.closed:
            raise RuntimeError("LISTEN connection is closed")
        return next(
            self._listen_connection.notifies(
                timeout=self._select_timeout,
                stop_after=1,
            ),
            None,
        )

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
            with psycopg.connect(
                user=config["username"],
                password=config["password"],
                dbname=config["database"],
                host=config["hostname"],
                port=int(config["port"]),
                autocommit=True,
            ) as conn:
                payload_json = json.dumps(payload)
                conn.execute("SELECT pg_notify(%s, %s)", (channel, payload_json))

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
