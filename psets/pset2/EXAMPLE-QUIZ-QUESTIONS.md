# Example Quiz Questions
Please see below for example quiz questions. We will have questions that are very similar to this for the quiz. 

## Question 1

In `classification_train.py`, students were asked to fill in the loss function for PathMNIST classification. The model predicts one of 9 tissue classes.

Which implementation correctly fills in the TODO?

**A.**
```python
criterion = nn.MSELoss()
````

**B.**

```python
criterion = nn.BCELoss()
```

**C.**

```python
criterion = nn.CrossEntropyLoss()
```

**D.**

```python
criterion = nn.L1Loss()
```

**Correct answer: C**

`nn.CrossEntropyLoss()` is appropriate for multi-class classification when the model produces a vector of logits for the 9 classes and `target` contains the correct class index. This matches the later code that obtains the predicted class with `output.max(1)`.

---

## Question 2

In `train_segmentation_epoch()` in `segmentation_train.py`, the starter code contains:

```python
optimizer.zero_grad()

# TODO: Implement the forward pass
predictions = 0

# TODO: Compute the loss
loss = 0

loss.backward()
optimizer.step()
```

Which replacement is correct?

**A.**

```python
predictions = model(masks)
loss = criterion(predictions, images)
```

**B.**

```python
predictions = model(images)
loss = criterion(predictions, masks)
```

**C.**

```python
predictions = criterion(images)
loss = model(predictions, masks)
```

**D.**

```python
predictions = model(images)
loss = criterion(images, masks)
```

**Correct answer: B**

The segmentation model receives the image and predicts a segmentation mask. The loss then compares the predicted mask against the ground-truth `masks`. The resulting scalar loss is differentiated by `loss.backward()`.

---

## Question 3

In `segmentation_models.py`, students implement `DiceLoss`. Assume `pred` and `target` contain corresponding predicted and ground-truth mask values.

Which expression correctly implements the core idea of Dice loss?

**A.**

```python
intersection = (pred * target).sum()
dice = (2 * intersection + self.smooth) / (
    pred.sum() + target.sum() + self.smooth
)
loss = 1 - dice
```

**B.**

```python
intersection = (pred + target).sum()
dice = intersection / (pred * target).sum()
loss = dice
```

**C.**

```python
intersection = (pred * target).sum()
dice = intersection / (pred.sum() - target.sum())
loss = 1 - dice
```

**D.**

```python
intersection = (pred == target).float().sum()
dice = intersection / pred.numel()
loss = dice
```

**Correct answer: A**

Dice measures overlap between the predicted and target masks:

$$
\text{Dice} =
\frac{2|\text{prediction} \cap \text{target}|}
{|\text{prediction}| + |\text{target}|}.
$$

Because training minimizes a loss, Dice loss is typically `1 - dice`. The `self.smooth` term prevents numerical problems when the denominator is near zero.