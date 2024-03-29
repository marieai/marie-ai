# MARIE_CLI_BEGIN

## autocomplete
if [[ ! -o interactive ]]; then
    return
fi

compctl -K _marie marieai

_marie() {
  local words completions
  read -cA words

  if [ "${#words}" -eq 2 ]; then
    completions="$(marieai commands)"
  else
    completions="$(marieai completions ${words[2,-2]})"
  fi

  reply=(${(ps:\n:)completions})
}

# session-wise fix
ulimit -n 4096
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# MARIE_CLI_END
