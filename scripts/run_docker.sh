docker run -it --ipc=host --network=host --group-add render \
        --privileged --security-opt seccomp=unconfined \
        --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
        --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        -v /remote/vast0/share-mv:/remote/vast0/share-mv \
        -v /remote/vast0/share-mv/tran/workspace:/workspace \
        -w /workspace/PGA-KD \
        -v /home/tran/.ssh:/root/.ssh \
        --name anhtt-exaone-dev \
        --entrypoint bash moreh-vllm:anhtt-dev