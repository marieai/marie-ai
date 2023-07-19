
## Reference
https://www.digitalocean.com/community/tutorials/how-to-use-vault-to-protect-sensitive-ansible-data
https://www.shellhacks.com/ansible-sudo-a-password-is-required/
https://github.com/priximmo
https://github.com/Pro-Tweaker/SEEDbox



PLAY RECAP ********************************************************************************************************************************************************************************************************
GPU-002                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-003                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-004                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-005                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-006                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-007                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-008                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-009                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-010                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-011                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-012                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-013                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-014                    : ok=1    changed=0    unreachable=0    failed=1    skipped=0    rescued=0    ignored=0   
GPU-015                    : ok=0    changed=0    unreachable=1    failed=0    skipped=0    rescued=0    ignored=0   
GPU-016                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-017                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-018                    : ok=0    changed=0    unreachable=1    failed=0    skipped=0    rescued=0    ignored=0   
GPU-019                    : ok=1    changed=0    unreachable=0    failed=1    skipped=0    rescued=0    ignored=0   
GPU-020                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-021                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-022                    : ok=2    changed=1    unreachable=0    failed=0    skipped=0    rescued=0    ignored=0   
GPU-023                    : ok=1    changed=0    unreachable=0    failed=1    skipped=0    rescued=0    ignored=0   
GPU-024                    : ok=1    changed=0    unreachable=0    failed=1    skipped=0    rescued=0    ignored=0   



changed: [GPU-006]
changed: [GPU-003]
changed: [GPU-002]
changed: [GPU-004]
changed: [GPU-005]
changed: [GPU-009]
changed: [GPU-008]
changed: [GPU-007]
changed: [GPU-011]
fatal: [GPU-014]: FAILED! => {"changed": false, "msg": "Error connecting: Error while fetching server API version: ('Connection aborted.', FileNotFoundError(2, 'No such file or directory'))"}
changed: [GPU-013]
changed: [GPU-012]
changed: [GPU-016]
fatal: [GPU-019]: FAILED! => {"changed": false, "msg": "Error connecting: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))"}
changed: [GPU-017]
fatal: [GPU-023]: FAILED! => {"changed": false, "msg": "Error connecting: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))"}
changed: [GPU-020]
fatal: [GPU-024]: FAILED! => {"changed": false, "msg": "Error connecting: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))"}
changed: [GPU-010]
changed: [GPU-021]
changed: [GPU-022]
