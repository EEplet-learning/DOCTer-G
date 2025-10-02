# DOCTer-G
Official implementation of DOCTer-G: A Generalizable Diagnosis Framework for Disorders of Consciousness Using EEG. This repository provides the source code, model architectures, and training scripts to reproduce the results reported in the paper.

## Dependencies

- Python 3.6+
- einops 0.6.0+
- mne 1.2.2+
- numpy 1.20.0+
- pandas 1.5.1+
- pycrostates 0.3.0+
- torch 1.13.0+
- scikit-learn 1.0.2+
- scipy 1.7.3+
- tqdm 4.64.1+

## Example

Here is a simple example of how to use the DOCTer-G framework:

```shell
python ./run_DOC/run_DOC.py --seed 2025 --trainseed 56  --exptype "DG" --model "MulTmp" --cuda 0 --param 0  --batch_size 512 --epochs 200 --p1 0.06 --p3 0.8 --th 4 --clip_value 1  --lr 0.0005 &
```

Or use a .sh script

```shell
nohup bash run.sh > log.txt 2>log.err &
```

## Notes

Thank you for visiting. The dataset cannot be made publicly available upon publication because it contains sensitive personal information. We will continue to improve this repository in the future.

## Contact

For questions, feedback, or suggestions, please contact us at:

- Email: [wyang2023@zju.edu.cn](mailto:wyang2023@zju.edu.cn)
- GitHub Issues: [Issues · EEplet/DOCTer](https://github.com/EEplet/DOCTer/issues)

## Citation

If you find our code is useful, please cite our paper.