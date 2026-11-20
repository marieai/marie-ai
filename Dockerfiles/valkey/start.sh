#!/bin/sh
set -eu
umask 077

# Keep private host files private while allowing the unprivileged server to read them.
for name in users.acl password; do
    test -s "/run/valkey-source/$name"
    cp "/run/valkey-source/$name" "/run/valkey/$name"
    chmod 600 "/run/valkey/$name"
done
digest_line="$(sha256sum /run/valkey/password)"
digest="${digest_line%% *}"
expected_acl="$(printf 'user default off\nuser marie on #%s ~* &* +@all\n' "$digest")"
if [ "$(cat /run/valkey/users.acl)" != "$expected_acl" ]; then
    echo >&2 "Valkey ACL does not match the configured credentials; refusing startup."
    exit 1
fi
chmod 700 /run/valkey
chown -R valkey:valkey /run/valkey

exec /usr/local/bin/docker-entrypoint.sh valkey-server /etc/valkey/valkey.conf
