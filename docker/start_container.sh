docker_run_command=$(cat run_container_w_gpu.sh)

docker_run_suffix=$(echo "$docker_run_command" | grep -oP "(?<=--name ).*"| awk '{print $1}')

docker start -ai "$docker_run_suffix"

