No container: this task is packaged for the `local` profile and runs on the host under
`harbor run -e p2c_task_re.harbor_env:LocalEnvironment` (see that module). Harbor discovers a task folder
only when it names a docker_image or has an environment/ folder, so p2c_task_re.stages.package writes this.
