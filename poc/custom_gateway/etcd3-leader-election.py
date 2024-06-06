import sys
import time
from threading import Event

import etcd3

__all__ = ["LeaderElection"]


class LeaderElection:
    def __init__(self, etcd_client, leader_key, my_id, lease_ttl):
        self.client = etcd_client
        self.leader_key = leader_key
        self.my_id = my_id
        self.lease_ttl = lease_ttl
        self.leader = None

    def elect_leader(self, leaderCb):
        """
        elect a leader.

        Args:
            leaderCb - leader callback function. If leader is changed, the
                       leaderCb will be called
        """
        while True:
            try:
                status, lease, self.leader = self._elect_leader()
                if self.leader is not None:
                    leaderCb(self.leader)
                if status:
                    self._refresh_lease(lease)
                else:
                    self._wait_leader_release()
                time.sleep(5)
            except Exception as ex:
                print(ex)

    def _elect_leader(self):
        try:
            lease = self.client.lease(self.lease_ttl)
            status, responses = self.client.transaction(
                compare=[self.client.transactions.version(self.leader_key) == 0],
                success=[
                    self.client.transactions.put(self.leader_key, self.my_id, lease)
                ],
                failure=[self.client.transactions.get(self.leader_key)],
            )
            if status:
                return status, lease, self.my_id
            elif len(responses) == 1 and len(responses[0]) == 1:
                return status, lease, responses[0][0][0]
        except Exception as ex:
            print(ex)
        return None, None, None

    def _refresh_lease(self, lease):
        """
        refresh the lease period
        """
        try:
            while True:
                lease.refresh()
                time.sleep(self.lease_ttl / 3.0 - 0.01)
        except (Exception, KeyboardInterrupt):
            pass
        finally:
            lease.revoke()

    def _wait_leader_release(self):
        """
        wait for the leader key deleted
        """
        leader_release_event = Event()

        def leader_delete_watch_cb(event):
            if isinstance(event, etcd3.events.DeleteEvent):
                leader_release_event.set()

        watch_id = None
        try:
            watch_id = self.client.add_watch_callback(
                self.leader_key, leader_delete_watch_cb
            )
            leader_release_event.wait()
        except:
            pass
        finally:
            if watch_id is not None:
                self.client.cancel_watch(watch_id)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="elect leader from etcd cluster")
    parser.add_argument(
        "--host",
        help="the etcd host, default = 127.0.0.1",
        required=False,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        help="the etcd port, default = 2379",
        required=False,
        default=2379,
        type=int,
    )
    parser.add_argument("--ca-cert", help="the etcd ca-cert", required=False)
    parser.add_argument("--cert-key", help="the etcd cert key", required=False)
    parser.add_argument("--cert-cert", help="the etcd cert", required=False)
    parser.add_argument("--leader-key", help="the election leader key", required=True)
    parser.add_argument(
        "--lease-ttl",
        help="the lease ttl in seconds, default is 10",
        required=False,
        default=10,
        type=int,
    )
    parser.add_argument("--my-id", help="my identifier", required=True)
    parser.add_argument(
        "--timeout",
        help="the etcd operation timeout in seconds, default is 2",
        required=False,
        type=int,
        default=2,
    )
    args = parser.parse_args()

    params = {"host": args.host, "port": args.port, "timeout": args.timeout}
    if args.ca_cert:
        params["ca_cert"] = args.ca_cert
    if args.cert_key:
        params["cert_key"] = args.cert_key
    if args.cert_cert:
        params["cert_cert"] = args.cert_cert

    client = etcd3.client(**params)

    leader_election = LeaderElection(
        client, args.leader_key, args.my_id, args.lease_ttl
    )

    def print_leader(leader):
        print("leader is %s" % leader)

    leader_election.elect_leader(print_leader)


if __name__ == '__main__':
    main()
