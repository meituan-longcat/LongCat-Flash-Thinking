# LongCat-Flash-Thinking Deployment Guide

We have implemented basic adaptations in both SGLang ([PR](https://github.com/sgl-project/sglang/pull/9824)) and vLLM ([PR](https://github.com/vllm-project/vllm/pull/23991)) to support the deployment of LongCat-Flash-Thinking.

Due to its size of 560 billion parameters (560B), LongCat-Flash-Thinking requires at least one node (e.g., 8xH20-141G) to host the model weights in FP8 format, and at least two nodes (e.g., 16xH800-80G) for BF16 weights. Detailed launch configurations are provided below.

The specific deployment is similar to our prior LongCat-Flash-Chat. We will list the deployment approach for SGLang and vLLM engines.

## SGLang

### Single-Node Deployment

The model can be served on a single node using a combination of Tensor Parallelism and Expert Parallelism.

```bash
python3 -m sglang.launch_server \
    --model meituan-longcat/LongCat-Flash-Thinking-FP8 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --enable-ep-moe \
    --tp 8
```

### Multi-Node Deployment

In a multi-node setup, Tensor Parallelism and Expert Parallelism are employed, with additional parallel strategies planned for future implementation.

```bash
python3 -m sglang.launch_server \
    --model meituan-longcat/LongCat-Flash-Thinking \
    --trust-remote-code \
    --attention-backend flashinfer \
    --enable-ep-moe \
    --tp 16 \
    --nnodes 2 \
    --node-rank $NODE_RANK \
    --dist-init-addr $MASTER_IP:5000
```

> [!Note]
> Replace $NODE_RANK and $MASTER_IP with the specific values for your cluster.

### Enabling Multi-Token Prediction (MTP)

To enable MTP with SGLang, add the following arguments to your launch command:

```bash
    --speculative-draft-model-path meituan-longcat/LongCat-Flash-Thinking \
    --speculative-algorithm NEXTN \
    --speculative-num-draft-tokens 2 \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1
```

## vLLM

While vLLM supports parallel strategies similar to SGLang, they use different syntax and parameter names for their launch commands.

### Single-Node Deployment

```bash
vllm serve meituan-longcat/LongCat-Flash-Thinking-FP8 \
    --trust-remote-code \
    --enable-expert-parallel \
    --tensor-parallel-size 8
```

### Multi-Node Deployment

```bash
# Node 0
vllm serve meituan-longcat/LongCat-Flash-Thinking \
   --trust-remote-code \
   --tensor-parallel-size 8 \
   --data-parallel-size 2 \
   --data-parallel-size-local 1 \
   --data-parallel-address $MASTER_IP \
   --data-parallel-rpc-port 13345

# Node 1
vllm serve meituan-longcat/LongCat-Flash-Thinking \
   --trust-remote-code \
   --tensor-parallel-size 8 \
   --headless \
   --data-parallel-size 2 \
   --data-parallel-size-local 1 \
   --data-parallel-start-rank 1 \
   --data-parallel-address $MASTER_IP \
   --data-parallel-rpc-port 13345
```

> [!Note]
> Only worker nodes (Node 1, 2, etc.) require the --data-parallel-start-rank flag.

### Enabling Multi-Token Prediction (MTP)

To enable MTP with vLLM, add the following arguments to your launch command:

```bash
    --speculative_config '{"model": "meituan-longcat/LongCat-Flash-Thinking", "num_speculative_tokens": 1, "method":"longcat_flash_mtp"}'
```

