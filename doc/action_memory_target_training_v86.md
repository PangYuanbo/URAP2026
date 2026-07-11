# Action Memory Target-Domain Training V86

## Controlled Results

| Dataset | Frozen NPS Memory | Target-Trained Memory | Training Gain | Existing Champion |
|---|---:|---:|---:|---:|
| ARD100 mAP@0.5 | 82.1190% | 82.2538% | +0.1349 points | 84.9004% |
| AOT AFDR | 83.1596% | 88.7711% | +5.6115 points | 89.9914% |

## Conclusion

- ARD100 target training contributes only a small improvement, so lack of training is not its main bottleneck.
- AOT target training contributes a large improvement, so domain-specific training is a major factor.
- Neither target-trained Cross-Attention model currently beats the mature legacy Action Bank champion.
- AOT x/y are center coordinates. The earlier converter used top-left coordinates; this controlled comparison uses corrected geometry.
- Test labels were not used for training or fusion selection.
