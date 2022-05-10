docker build -t $1 --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
