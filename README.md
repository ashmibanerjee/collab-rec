# Collab-Rec: An LLM-based Agentic Framework for Tourism Recommendations

**Authors:** Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo

**Paper URL:** [https://arxiv.org/pdf/2508.15030](https://arxiv.org/pdf/2508.15030)

**Abstract:**  
We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agent s— **Personalization**, **Popularity**, and **Sustainability** — generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses.
Experiments on European city queries demonstrate that \sysname{} enhances diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that are often overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems
## Features
- Multi-agent design addressing personalization, popularity, and sustainability
- Iterative negotiation for balanced, context-aware recommendations
- Reduced popularity bias and surfacing of lesser-visited locales
- Reproducible pipeline

## Artifacts
The repository includes:
- Code and scripts
- Example prompts for each agent
- Datasets used in experiments

## Citation
If you use this work, please cite the paper:

```
@article{banerjee2025collab,
  title={Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism},
  author={Banerjee, Ashmi and Aisyah, Fitri Nur and Satish, Adithi and W{\"o}rndl, Wolfgang and Deldjoo, Yashar},
  journal={arXiv preprint arXiv:2508.15030},
  year={2025}}
```
