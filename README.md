# Transformer from Scratch (PyTorch)
> **Attention Is All You Need** 논문을 기반으로 Transformer를 PyTorch로 처음부터 구현하고, 핵심 개념을 6단계로 설명합니다.

<p align="center">
  <img src="transformer_architecture.jpg" alt="Transformer Architecture" width="350">
</p>

<p align="center">
  <em>Figure. High-level overview of the Transformer architecture</em>
</p>

---

## Overview
이 레포지토리는 Vaswani et al.(2017)의 *Attention Is All You Need* 논문을 바탕으로  
Transformer의 핵심 구성 요소를 **from scratch** 방식으로 PyTorch에서 구현한 프로젝트입니다.

- PyTorch의 고수준 `nn.Transformer`를 사용하지 않음
- 논문 수식과 구조에 최대한 일치하도록 구현
- 구현 → 개념 → 수식 흐름으로 설명

---

## 6-Step Explanation Roadmap
Transformer를 다음 **6단계**로 나누어 설명합니다.

1. **Input Embedding & Scaling**
2. **Positional Encoding**
3. **Scaled Dot-Product Attention**
4. **Multi-Head Attention**
5. **Add & Norm (Residual + LayerNorm)**
6. **Position-wise Feed-Forward Network**

---

## Step 1. Input Embedding & Scaling
### What
- 토큰 ID를 고정 차원 벡터(`d_model`)로 변환
- 임베딩 결과에 `sqrt(d_model)` 스케일링 적용

### Why
- 임베딩과 positional encoding의 스케일을 맞추기 위함
- 초기 학습 안정성 향상

### Shape
