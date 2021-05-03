# TechSoc IHC Deep Learning Hack 2021: Track 2 - Paper Abstract Generation
**Team**: Nishant Prabhu, Shrisudhan G.

Our solution for DL Hack 2021 Track 2 by Analytics Club, IITM. Network to generate abstracts for scientific papers given their title.

# Trained model
Pretrained BART finetuned on given data in teacher-forcing mode: add link here

# Usage
Code to train the best model was written by Nishant Prabhu. To re-train the model, run the following command inside `main`:

```
python3 main.py --config 'configs/main.yaml' --task 'train'
```

To perform inference on the test set with the trained model, ensure that the data provided in the contest is available in a directory named `track2_data` on the same directory level as `main`. Then, run the following command inside `main`:

```
python3 main.py --config 'configs/main.yaml' --load '/path/to/dir/containing/best_model.ckpt' --task 'test'
```

Once the model's predictions are generated, `submit_track2.py` can be called with the output file as CLI arguments to generated the embedding file, as required by Kaggle.
