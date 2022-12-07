# MARIE_CLI_BEGIN

## autocomplete
function __fish_marie_needs_command
  set cmd (commandline -opc)
  if [ (count $cmd) -eq 1 -a $cmd[1] = 'marie' ]
    return 0
  end
  return 1
end

function __fish_marie_using_command
  set cmd (commandline -opc)
  if [ (count $cmd) -gt 1 ]
    if [ $argv[1] = $cmd[2] ]
      return 0
    end
  end
  return 1
end

complete -f -c marie -n '__fish_marie_needs_command' -a '(marie commands)'
for cmd in (marie commands)
  complete -f -c marie -n "__fish_marie_using_command $cmd" -a \
    "(marie completions (commandline -opc)[2..-1])"
end

# session-wise fix
ulimit -n 4096
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# MARIE_CLI_END
