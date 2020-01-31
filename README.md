# SFkNNI
Separate Features kNN based method for missing value imputation

This is a C# implementation of my Separate Features kNN Algorithm modified for missing values imputation.
This is a kNN classifier, which processes each feature of the original entity separately. This allows it to operate on non-fully defined entities - both for reference (learning) and query (classification). I have found it to achieve good classification results on multidimensional data (6+ features) with a significant amount (up to 35%) of randomly placed missing (undefined) values. It doesn't require any missing value pre-processing, like filtering, or substituting.
Here it is used to impute missing values for other classification algorithms.
