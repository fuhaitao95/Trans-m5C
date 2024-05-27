# Trans-m5C: A transformer-based model for predicting 5-methylcytosine (m5C) sites
Code for the paper "Trans-m5C: A transformer-based model for predicting 5-methylcytosine (m5C) sites".

Developer: Fu Haitao, School of Artificial Intelligence, Hubei University, Wuhan 430070, China; Ding Zewen, University of Edinburgh, Edinburgh, EH89YL, Scotland; Wang Wen, University of Edinburgh, Edinburgh, EH89YL, Scotland.

## Introduction
5-Methylcytosine (m5C) plays a pivotal role in various RNA metabolic processes, including RNA localization, stability, and translation. Current high-throughput sequencing technologies for m5C site identification are resource-intensive in terms of cost, labor, and time. As such, there is a pressing need for efficient computational approaches. Many existing computational methods rely on intricate hand-crafted features, requiring unavailable features, often leading to suboptimal prediction accuracy. Addressing these challenges, we introduce a novel deep-learning method, Transm5C. We first categorize m5C sites into NSUN2-dependent and NSUN6-dependent types for independent feature extraction. Subsequently, a meticulously crafted transformer neural network is employed to distill global features. The prediction of m5C sites is then accomplished using a discriminator built from multiple fully connected layers. A rigorous evaluation for the performance of Transm5C on experimentally validated m5C data from human and mouse species reveals that our method offers a competitive edge over both baseline and existing methodologies.

## Requirements
python==3.7.1

pytorch 1.6.0

sklearn==0.24.2

numpy==1.21.2

pandas==1.2.0

## Usage
```shell
# example for human species
python m5_main.py --config ../Config/Liu2022Developmental_Human_exon_YN_ctcca/config_general_TraValTes_75_8497.json
# example for mouse species
python m5_main.py --config ../Config/Liu2022Developmental_Mouse_transcript_YN_ctcca/config_general_TraValTes_31_8424.json
```

You can change the `Dataset` file and the `config` file for your own dataset.

## Contact
- Please feel free to contact us if you need any help: 2157523736@qq.com.
- **Attention**: Only real name emails will be replied. Please provide as much detail as possible about the problem you are experiencing.
- **注意**：只回复实名电子邮件。请尽可能详细地描述您遇到的问题，可以附上截图等。