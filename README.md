
# Zero-shot-PolSAR-target-recognition
This is the official implementation of ***POlSAR-ZSL***, a PolSAR recognition method. For more details, Comming Soon!

**Zero-shot-PolSAR-target-recognition [[Paper]]()**  <br />
Xiaojing Yang<br />

## Installation

Please refer to [install.md](docs/install.md) for installation.


## Getting Started
## Results

<table><tr><th rowspan="2">Baseline</th><th>ZSL</th><th colspan="3">GZSL</th></tr>
<tr><td>T<sub>1</sub></td><td>S</td><td>U</td><td>H</td></tr>
<tr><td>TF-VAEGAN</td><td>0.7383</td><td>0.7886</td><td>0.4833</td><td>0.5993</td></tr>
<tr><td>Ours(TF)</td><td>0.9222</td><td>0.8390</td><td>0.6861</td><td>0.7549</td></tr>
<tr><td>CE-GZSL</td><td>0.8217</td><td>0.7734</td><td>0.4038</td><td>0.5305</td></tr>
<tr><td>Ours(CE)</td><td>0.9250</td><td>0.7679</td><td>0.6814</td><td>0.7221</td></tr>
</table>


## Evaluate
1.GOTCHA on Ours(CE-GZSL) Test
```shell
python Predict.py --cuda --preprocessing
```

