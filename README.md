# LongCat-Flash-Thinking

<div align="center">
  <img src="figures/longcat_logo.svg" width="45%" alt="LongCat-Flash" />
</div>
<hr>


<div align="center" style="line-height: 1;">
  <a href="https://longcat.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-LongCat--Flash--Thinking-ADFF2F?color=29E154&logoColor=white"  fill-opacity="1" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/meituan-longcat" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/meituan-longcat/LongCat-Flash-Thinking/blob/main/figures/wechat_official_accounts.png" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://x.com/Meituan_LongCat" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="./tech_report.pdf"><b>Tech Report</b>&nbsp;📄</a>
</p>



## Model Introduction

We introduce and release **LongCat-Flash-Thinking**, which is a powerful and efficient large reasoning model (LRM) with 560 billion total parameters, featuring an innovative Mixture-of-Experts (MoE) architecture. The model incorporates a dynamic computation mechanism that activates 18.6B∼31.3B parameters (averaging∼27B) based on contextual demands, optimizing both computational efficiency and performance. LongCat-Flash-Thinking is developed by our DORA system, which is an efficient distributed RL framework that supports asynchronous training and flexible accelerator usage to ensure stability and efficiency. Our comprehensive data curation and domain-parallel training recipe ensures stable and efficient training. In addition to general reasoning, the model is also equipped with techniques of formal reasoning and agentic reasoning, advancing the LRMs' reasoning ability on diverse complex tasks such as mathematics, logic, programming, automatic theorem proving, and tool use.

Specifically, the development of LongCat-Flash-Thinking follows a two-phase pipeline:
- **Long CoT Cold-Start Training**: This phase aims to cultivate the model's foundational reasoning abilities. 
This begins with a curriculum learning strategy during mid-training to bolster intrinsic capabilities, followed by a SFT stage on reasoning-intensive and agentic data to prepare the model for advanced learning.
- **Large-Scale RL**: The second phase scales up this potential through an efficient RL framework, built upon our Dynamic Orchestration for Asynchronous Rollout (DORA) system for industrial-scale asynchronous training. 
To address the stability challenges in asynchronous RL training, we adapt and extend the GRPO algorithm for a robust exploration-exploitation balance. A key innovation in this phase is our domain-parallel training scheme, which simultaneously optimizes the model across distinct domains and subsequently merges the resulting domain-expert models into a fused model. Finally, we perform a general RL stage to further refine the fused model and enhance its robustness, safety, and human alignment ability. 


![](./figures/parallel_training_pipeline.svg)

### Key Features

#### 🌟 Domain-Parallel RL Training Methodology

To overcome the instability of traditional mixed-domain RL training, LongCat-Flash-Thinking incorporates a domain-parallel training scheme that decouples optimization across STEM, coding, and agentic tasks. 
This approach not only stabilizes training, but also allows to fuse the resulting domain-expert models into a nearly Pareto-optimal final model that excels across all specialties.

#### 🌟 Pioneering RL Infrastructure

LongCat-Flash-Thinking is built upon our self-designed DORA system. 
The main motivation is to optimize long-tail generation by leveraging multiple old versions of the Actor model through streaming rollout while keeping sampling consistency.
DORA system consists of two core components, such as elastic colocation and multi-version asynchronous pipeline. These components aim to enhance training efficiency, ensure policy consistency per sample, and further enable efficient KV-cache reuse, facilitating stable and scalable training on tens of thousands of accelerators.

#### 🌟 Advancing Formal Reasoning and Agentic Reasoning

In addition to general reasoning (e.g., mathematics, logic, coding, instruction-following, etc.), LongCat-Flash-Thinking also emphasizes two other critical capabilities.
- **Formal Reasoning**: LongCat-Flash-Thinking can solve complex formal reasoning tasks, e.g., automatic theorem proving. To help realize this potential and empower researchers, we introduce significant enhancements to our model's formal reasoning capabilities.
To achieve this, we introduce a novel expert iteration framework for careful data synthesis, involving statement formalization, iterative proof synthesis, and syntax/consistency filtering.
- **Agentic Reasoning**: LongCat-Flash-Thinking can adaptively utilize provided tools to solve complex reasoning tasks. To reach this goal, we introduce a dual-path reasoning approach to identify and retain high-quality queries that genuinely require tool assistance, thereby fostering the development of robust agentic abilities.
After high-value query selection, we synthesize corresponding high-quality
solution trajectories based on a versatile environment with diverse tool APIs,
including MCP servers and simulated tools for both single and multi-turn interactions.  

For more details, please refer to the comprehensive [**LongCat-Flash-Thinking Technical Report**](https://github.com/meituan-longcat/LongCat-Flash-Thinking/blob/main/tech_report.pdf).

## Evaluation Results

| **Benchmark** | DeepSeek-V3.1-Thinking | Qwen3-235B-A22B-Thinking-2507 | GLM-4.5 | OpenAI-o3 | Gemini2.5-Pro | GPT-5-Thinking | LongCat-Flash-Thinking |
|---------------|-------------------------|------------------------------|--------|-----------|---------------|----------------|-------------------------|
| Architecture  | MoE                     | MoE                          | MoE    | -         | -             | -              | MoE                     |
| \# Total Params | 671B                   | 235B                         | 355B   | -         | -             | -              | 560B                    |
| \# Activated Params | 37B                | 22B                          | 32B    | -         | -             | -              | 27B                     |
| **General QA** |                        |                              |        |           |               |                |                         |
| MMLU-Pro<sub>(acc)</sub>    | 84.4                   | 84.4                         | 81.5   | 85.3  | 86.7       | 84.5           | 82.6                    |
| MMLU-Redux<sub>(acc)</sub>  | 90.5                   | 91.4                         | 89.9   | 93.1         | 90.1            | 92.6  | 89.3                    |
| **Alignment** |                         |                              |        |           |               |                |                         |
| IFEval<sub>(strict prompt)</sub> | 86.3          | 89.3                         | 85.4   | 90.2      | 92.4  | 92.8        | 86.9                    |
| Arena-Hard<sub>(hard prompt gemini)</sub> | 57.1 | 74.5                    | 67.7   | 87.1  | 87.1  | 87.7        | 69.9                   |
| **Mathematical Reasoning** |             |                              |        |           |               |                |                         |
| MATH500<sub>(Mean@1)</sub>  | 98.8                   | 99.6                    | 95.4   | 98.4      | 98.0          | 99.2  | 99.2        |
| HMMT25<sub>(Mean@32)</sub>  | 80.4                   | 83.8            | 76.3   | 71.9      | 79.3          | 84.8       | 83.7                    |
| AIME24<sub>(Mean@32)</sub>  | 93.9               | 93.9                    | 89.3   | 91.6* | 90.7   | 92.0  | 93.3        |
| AIME25<sub>(Mean@32)</sub>  | 87.9                   | 92.5                         | 85.5   | 88.9*  | 89.2   | 94.6*  | 90.6        |
| BeyondAIME<sub>(Mean@10)</sub> | 71.8             | 71.5                         | 66.0   | 63.2      | 63.0          | 70.0          | 69.5        |
| **General Reasoning** |                  |                              |        |           |               |                |                         |
| GPQA-Diamond<sub>(Mean@16)</sub> | 84.2         | 80.4                         | 78.3   | 81.9      | 84.0  | 84.4       | 81.5                   |
| ZebraLogic<sub>(Mean@1)</sub> | 96.1            | 97.5                    | 90.9   | 94.3      | 92.4          | 92.7          | 95.5       |
| Sudoku-Bench<sub>(Mean@1)</sub> | 1.0           | 2.0                         | 1.0    | 70.0  | 0.0           | 63.0  | 56.0                   |
| ARC-AGI<sub>(Mean@1)</sub>    | 37.5                   | 45.3                         | 21.41  | 47.3  | 46.8          | 59.0       | 50.3                   |
| **Coding** |                         |                              |        |           |               |                |                         |
| LiveCodeBench<sub>(Mean@4)</sub> | 73.5      | 75.4                         | 61.1   | 76.2      | 74.2          | 80.6       | 79.4       |
| OJBench<sub>(Mean@1)</sub>    | 33.6                   | 32.1                         | 19.0   | 38.4      | 41.6       | 34.1          | 40.7       |
| **Agentic Tool Using** |                 |                              |        |           |               |                |                         |
| SWE-Bench<sub>(Pass@1)</sub> | 66.0* | 34.4                       | 64.2* | 69.1*  | 59.6*  | 74.9*  | 59.4                   |
| BFCL V3<sub>(full)</sub>      | 55.4                   | 75.7                         | 79.1   | 72.4*  | 63.2          | 60.1          | 74.4              |
| τ²-Bench-Retail<sub>(Mean@4)</sub> | 65.4 | 68.2                         | 69.3   | 72.8      | 70.9          | 81.1*  | 71.5       |
| τ²-Bench-Airline<sub>(Mean@4)</sub> | 44.0 | 58.0                        | 66.0   | 62.5      | 58.0          | 62.6*  | 67.5       |
| τ²-Bench-Telecom<sub>(Mean@4)</sub> | 23.7 | 47.3                       | 56.1   | 67.5      | 38.3          | 96.7*  | 83.1       |
| VitaBench | 13.5                       | 21.5                         | 26.8   | 35.3      | 24.3          | 29.3          | 29.5                   |
| **Formal Theorem Proving** |            |                              |        |           |               |                |                         |
| MiniF2F-Test<sub>(Pass@1)</sub> | 49.6       | 11.9                         | 10.9   | 15.2      | 13.9          | 21.4          | 67.6              |
| MiniF2F-Test<sub>(Pass@8)</sub> | 74.4       | 20.9                         | 22.1   | 29.6      | 29.4          | 39.7          | 79.4              |
| MiniF2F-Test<sub>(Pass@32)</sub> | 79.5      | 26.6                        | 27.0   | 37.7      | 41.8          | 51.2          | 81.6              |
| **Safety** |                             |                              |        |           |               |                |                         |
| Harmful | 79.2                            | 84.3            | 70.4   | 64.8      | 44.3          | 56.8          | 93.7               |
| Criminal | 89.7                          | 92.7                         | 88.8   | 85.7      | 77.4          | 87.3          | 97.1               |
| Misinformation | 81.1        | 80.9                         | 67.1   | 42.7      | 31.0          | 41.9          | 93.0               |
| Privacy | 96.2                           | 100.0                    | 97.6   | 100.0 | 95.0          | 98.8  | 98.8        |

Note:
- Values marked with * are sourced from other public reports.
- The inference parameters of our LongCat-Flash-Thinking are set as `temperature=1.0`, `topk=-1`, and `topp=0.95`. 


## Quick Start

### Chat Template
The details of our chat template are provided in the `tokenizer_config.json` file. Below are some examples.

#### First-Turn

With the following prefix, LongCat-Flash can generate responses corresponding to user queries:

```
[Round 0] USER:{query} /think_on ASSISTANT:
```

When a system prompt is specified, the prefix will take the following format:

```
SYSTEM:{system_prompt} [Round 0] USER:{query} /think_on ASSISTANT:
```

#### Multi-Turn

In multi-turn scenarios, the prefix is constructed by concatenating the context with the latest user query:
```
SYSTEM:{system_prompt} [Round 0] USER:{query} /think_on ASSISTANT:{response}... [Round N-1] USER:{query} /think_on ASSISTANT:{response} [Round N] USER:{query} /think_on ASSISTANT:
```

Here, $N$ denotes the $N$ -th round of user queries, with indexing starting from zero.

#### ToolCall

LongCat-Flash supports tool calling in the following format:
```
{tool_description}

## Messages
SYSTEM:{system_prompt} [Round 0] USER:{query} /think_on ASSISTANT:
```



The tool_description is:
```markdown
## Tools
You have access to the following tools: 

### Tool namespace: function

#### Tool name: {func.name}

Description: {func.description}

InputSchema: 
{json.dumps(func.parameters, indent=2)}

**Note**: For each function call, return a json object with function name and arguments within <longcat_tool_call></longcat_tool_call> XML tags as follows:
<longcat_tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</longcat_tool_call>
When multiple functions need to be called simultaneously, each function call should be wrapped in its own <longcat_tool_call> tag and placed consecutively. For example:
<longcat_tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</longcat_tool_call><longcat_tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</longcat_tool_call>
```

#### Mathematical Reasoning
We recommend adding the following instructions when solving mathematical or other STEM-related reasoning tasks, so that the output results can be located for evaluation.

```text
[Round 0] USER:{problem}
Please reason step by step, and put your final answer within \\boxed{}. /think_on ASSISTANT:
```



#### Formal Reasoning

LongCat-Flash-Thinking also supports formal reasoning, like automatic theorem proving (ATP). The specific template is:

```text
[Round 0] USER:Think about and solve the following problem step by step in Lean 4.
# Problem:{problem}

# Formal statement:{formal_statement}
 /think_on ASSISTANT:
```



## Deployment
We have implemented basic adaptations in both SGLang and vLLM to support the deployment of LongCat-Flash-Thinking. Please refer to the [Deployment Guide](docs/deployment_guide.md) for detailed deployment instructions.

## Chat Website
You can chat with LongCat-Flash-Thinking on our official website: [https://longcat.ai](https://longcat.ai).
Please turn on the button "Think" ("深度思考" in Chinese) before submitting your request.

## License Agreement

The **model weights** are released under the **MIT License**. 

Any contributions to this repository are licensed under the MIT License, unless otherwise stated. This license does not grant any rights to use Meituan trademarks or patents. 

See the [LICENSE](LICENSE) file for the full license text.

## Usage Considerations 
This model has not been specifically designed or comprehensively evaluated for every possible downstream application. 

Developers should take into account the known limitations of large language models, including performance variations across different languages, and carefully assess accuracy, safety, and fairness before deploying the model in sensitive or high-risk scenarios. 
It is the responsibility of developers and downstream users to understand and comply with all applicable laws and regulations relevant to their use case, including but not limited to data protection, privacy, and content safety requirements. 

Nothing in this Model Card should be interpreted as altering or restricting the terms of the MIT License under which the model is released. 

## Citation
We kindly encourage citation of our work if you find it useful.

```
@misc{meituan2025longcatflashthinkingtechnicalreport, 
    title={LongCat-Flash-Thinking Technical Report}, 
    author={Meituan}, 
    year={2025}, 
    eprint={2509.18883}, 
    archivePrefix={arXiv}, 
    primaryClass={cs.AI}, 
    url={https://arxiv.org/abs/2509.18883}, 
}
```

## Contact
Please contact us at <a href="mailto:longcat-team@meituan.com">longcat-team@meituan.com</a> or join our WeChat Group if you have any questions.

#### WeChat Group
<img src=figures/wechat_qrcode.png width="200px">
