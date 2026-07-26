
ps aux | grep marie | grep -v grep | awk '{print $2}' | xargs -r kill -9

# pkill -KILL -f '/home/greg/dev/marieai/marie-ai/.venv/bin/python'
