# ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization in Audio-Language Models (INTERSPEECH'26)

> [**ZEBRA: Zero-Shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization in Audio-Language Models**](https://arxiv.org/abs/2606.31587)<br><br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ) and [Mohammad Yaqub](https://scholar.google.com/citations?user=FXJzma8AAAAJ)

<!-- [![page](https://img.shields.io/badge/Project-Page-F9D371)](https://asif-hanif.github.io/trojanwave/) -->
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2606.31587)




<hr />

| ![main figure](/media/zebra.png)|
|:--| 
| **ZEBRA**<p align="justify">ZEBRA approach operates on top of existing prompt learning methods to bridge the base-to-novel generalization gap, preserving zero-shot transferability while benefiting from supervised adaptation through few-shot prompt learning. ZEBRA introduces no additional learnable parameters to existing prompt learning methods and incurs negligible computational overhead.</p> |

</br>


</br>
<hr />
</br>

> **Abstract** <p align="justify"><i>
Audio-Language Models (ALMs) achieve strong zero-shot performance by aligning audio with textual class descriptions. Although prompt learning improves accuracy on base classes through few-shot supervised adaptation, we observe a critical trade-off: it often degrades performance on novel classes, sometimes falling below zero-shot accuracy. This exposes a base-to-novel generalization gap in prompt learning for ALMs. To address this issue, we propose ZEBRA (Zero-shot Entropy-Regularized Prompt Learning for Base-to-Novel Generalization), a plug-and-play framework that fuses zero-shot logits with prompt-learning logits, and employs self-entropy regularization to reduce overfitting to base classes. Experiments across multiple audio classification datasets show that ZEBRA consistently improves novel-class performance while maintaining strong base accuracy, significantly reducing the base-to-novel gap compared to standard prompt learning. 
<br><br>
</i></p>

> <b>TLDR:</b> ZEBRA is a plug-and-play framework for Audio-Language Models that fixes the issue where prompt learning improves known-class accuracy but hurts performance on unseen classes. By fusing zero-shot predictions and applying self-entropy regularization, it significantly boosts accuracy on new categories without requiring any extra learnable parameters.

<br><br>
