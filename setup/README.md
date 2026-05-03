# Setup helpers

## `local_pubkey.txt` — pranav's laptop SSH public key

Used to enable password-less rsync from pranav's laptop to the iDRAC
server (10.29.8.249) where experiments run. To install on a fresh
server (one-time):

```bash
git pull
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat setup/local_pubkey.txt >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

The corresponding private key lives only on the laptop at
`~/.ssh/id_ed25519` and is never committed.
