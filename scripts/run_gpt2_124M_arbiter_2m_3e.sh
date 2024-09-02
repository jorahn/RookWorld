# GPT-2 (124M) repro on FineWeb
# 124M parameter model on 10B tokens
# => 6 * 124e6 * 10e9 = 7.44e18 ~= 7e18 capability model
# 18,865 steps of 524,288 tokens/step
# on 8X A100 80GB SXM ($14/hr) steps in ~300ms/iter
# => training time 18,865 * 300ms = 94.3 min ~= $20

make train_gpt2cu USE_CUDNN=1
out_dir="log_gpt2_124M_arbiter_2m_3e"
done_file="$out_dir/DONE_00012762"

# in case the training stalls or crashes, loop to resume (-y 1)
while true; do

    # exit condition is that optimization has finished
    if [ -f "$done_file" ]; then
        echo "File $done_file exists. Exiting the loop."
        break
    fi

    # run python dev/data/arbiter.py to prepro data
    mpirun -np 2 ./train_gpt2cu \
                -i "dev/data/arbiter/arbiter_train_*.bin" \
                -j "dev/data/arbiter/arbiter_val_*.bin" \
                -o $out_dir \
                -v 100 -s 5000 -g 144 \
                -n 5000 \
                -h 0 \
                -b 32 -t 1024 \
                -d 65536 \
                -r 0 \
                -z 0 \
                -c 0.1 \
                -l 0.0004 \
                -q 0.0 \
                -u 500 \
                -y 1 \
                -x 12762 \
                -e "d12"

    sleep 1
done
