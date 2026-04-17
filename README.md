# telecom-learning-curves

## 1. Bias vs Variance

The learning curve shows that both training and validation scores are low and converge as the training size increases. The gap between them is relatively small, especially at larger dataset sizes.

This indicates that the model is suffering from high bias (underfitting) rather than high variance. The model is too simple to capture the underlying patterns in the data, which results in poor performance on both training and validation sets.

---

## 2. Would more data help?

Collecting more data is unlikely to significantly improve performance.

Evidence:

* The training and validation curves are already close together
* Both scores remain consistently low

This suggests the model has already learned as much as it can given its current capacity. Adding more data will not fix underfitting.

---

## 3. Would increasing model complexity help?

Yes, increasing model complexity would likely help.

Since the model is underfitting, a more flexible model could better capture relationships in the data. Possible improvements include:

* Adding polynomial features
* Using a more complex model ( Random Forest, Gradient Boosting)
* Reducing regularization strength in logistic regression

---

## 4. Recommended next step

The best next step is to increase model complexity.

Specifically:

* Try a more expressive model like Random Forest or Gradient Boosting
* Or enhance logistic regression with feature engineering (polynomial features, interactions)

This should help reduce bias and improve both training and validation performance.
