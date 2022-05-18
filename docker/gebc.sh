nvidia-docker run -it -h gebc \
	-p 1265:1265 \
	--ipc=host \
	--name gebc \
	-v /m2:/projects \
	-v /hdd/datasets:/datasets \
	tootouch/gebc bash
