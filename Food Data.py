#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd


# In[16]:


df=pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\Desktop\\mcdonalds.csv')


# In[17]:


df.columns


# In[18]:


df.shape


# In[19]:


df.head(3)


# In[127]:


#PCA


# In[20]:


MD_x = df.iloc[:, :11].values
MD_x = (MD_x == "Yes").astype(int)
col_means = np.round(MD_x.mean(axis=0), 2)

print(col_means)


# In[21]:


from sklearn.decomposition import PCA

pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Creating a DataFrame to summarize the PCA results
explained_variance = pca.explained_variance_ratio_
components = pca.components_

# Summary
summary_df = pd.DataFrame({
    'Explained Variance Ratio': explained_variance,
    'Principal Components': components.tolist()
})

print(summary_df)


# In[22]:


from sklearn.decomposition import PCA

pca = PCA()
MD_pca = pca.fit(MD_x)

# Extracting standard deviations (square roots of the eigenvalues)
standard_deviations = pca.singular_values_

# Printing the standard deviations with 1 decimal place
print("Standard deviations (1, .., p={}):".format(MD_x.shape[1]))
print(np.round(standard_deviations, 1))


# In[23]:


pca = PCA()
MD_pca = pca.fit(MD_x)

# Extracting the rotation matrix (principal components)
rotation_matrix = pca.components_

# Printing the rotation matrix
print("Rotation (n x k) = ({} x {}):".format(rotation_matrix.shape[0], rotation_matrix.shape[1]))
print(rotation_matrix)


# In[24]:


pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Plotting the PCA results
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of McDonalds Data')
plt.grid(True)

# Projecting axes (similar to projAxes in R)
for i, (comp1, comp2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, comp1, comp2, color='r', alpha=0.5)
    plt.text(comp1 * 1.15, comp2 * 1.15, f"Var{i+1}", color='g', ha='center', va='center')

plt.show()


# In[25]:


# Clustering


# In[26]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Set seed for reproducibility
np.random.seed(1234)

# Function to perform clustering and relabeling
def step_flexclust(data, k_range, nrep=10):
    best_models = []
    for k in k_range:
        best_inertia = np.inf
        best_model = None
        for _ in range(nrep):
            model = KMeans(n_clusters=k, n_init=1, random_state=np.random.randint(0, 10000))
            model.fit(data)
            if model.inertia_ < best_inertia:
                best_inertia = model.inertia_
                best_model = model
        best_models.append(best_model)
    return best_models

# Perform clustering
k_range = range(2, 9)
MD_km28 = step_flexclust(MD_x, k_range, nrep=10)

# Relabel clusters to ensure consistency
def relabel(models):
    # Assuming relabeling is based on some criteria, here we just return the models as is
    return models

MD_km28 = relabel(MD_km28)

# Print the best model for each k
for i, model in enumerate(MD_km28, start=2):
    print(f"Best model for k={i}:")
    print(f"Inertia: {model.inertia_}")
    print(f"Cluster centers:\n{model.cluster_centers_}\n")


# In[27]:


import matplotlib.pyplot as plt

# Assuming MD_km28 is a list of KMeans models as defined in the previous example
inertia_values = [model.inertia_ for model in MD_km28]
num_segments = range(2, 9)

plt.plot(num_segments, inertia_values, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Inertia')
plt.title('Clustering Results')
plt.grid(True)
plt.show()


# In[28]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import resample

# Set seed for reproducibility
np.random.seed(1234)

# Function to perform bootstrapping and clustering
def boot_flexclust(data, k_range, nrep=10, nboot=100):
    boot_results = []
    for k in k_range:
        best_models = []
        for _ in range(nrep):
            model = KMeans(n_clusters=k, n_init=1, random_state=np.random.randint(0, 10000))
            model.fit(data)
            best_models.append(model)
        
        # Bootstrapping
        boot_inertia = []
        for _ in range(nboot):
            sample_data = resample(data)
            model = KMeans(n_clusters=k, n_init=1, random_state=np.random.randint(0, 10000))
            model.fit(sample_data)
            boot_inertia.append(model.inertia_)
        
        boot_results.append({
            'k': k,
            'models': best_models,
            'boot_inertia': boot_inertia
        })
    return boot_results

# Perform bootstrapping and clustering
k_range = range(2, 9)
MD_b28 = boot_flexclust(MD_x, k_range, nrep=10, nboot=100)

# Print the bootstrapping results
for result in MD_b28:
    print(f"Number of clusters: {result['k']}")
    print(f"Bootstrap inertia (mean): {np.mean(result['boot_inertia']):.2f}")
    print(f"Bootstrap inertia (std): {np.std(result['boot_inertia']):.2f}\n")


# In[46]:


import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Assuming MD_b28 is a list of dictionaries as defined in the previous example
# and each dictionary contains 'models' and 'boot_inertia'

# Function to calculate adjusted Rand index for each bootstrap sample
def calculate_ari(models, data):
    ari_scores = []
    for model in models:
        labels = model.predict(data)
        ari = adjusted_rand_score(labels, labels)  # Adjusted Rand index with itself
        ari_scores.append(ari)
    return ari_scores

# Calculate ARI for each k
ari_results = []
for result in MD_b28:
    k = result['k']
    models = result['models']
    ari_scores = calculate_ari(models, MD_x)
    ari_results.append((k, np.mean(ari_scores)))

# Extracting number of segments and ARI values
num_segments = [result[0] for result in ari_results]
ari_values = [result[1] for result in ari_results]

# Plotting the results
plt.plot(num_segments, ari_values, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Bootstrapping Clustering Results')
plt.grid(True)
plt.show()


# In[47]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming MD_km28 is a list of KMeans model
# and MD_x is the data matrix

# Extracting the cluster labels for k=4
kmeans_model = MD_km28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)
labels = kmeans_model.labels_

# Plotting the histogram
plt.hist(labels, bins=np.arange(0, 2, 0.1), edgecolor='black', color='grey')
plt.xlim(0, 1)
plt.xlabel('Cluster Labels')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Labels for k=4')
plt.grid(True)
plt.show()


# In[48]:


import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Assuming MD_b28 is a list of dictionaries 
# and each dictionary contains 'models' and 'boot_inertia'

# Function to calculate adjusted Rand index for each bootstrap sample
def calculate_ari(models, data):
    ari_scores = []
    for model in models:
        labels = model.predict(data)
        ari = adjusted_rand_score(labels, labels)  # Adjusted Rand index with itself
        ari_scores.append(ari)
    return ari_scores
# Calculate ARI for each k
ari_results = []
for result in MD_b28:
    k = result['k']
    models = result['models']
    ari_scores = calculate_ari(models, MD_x)
    ari_results.append((k, np.mean(ari_scores)))

# Extracting number of segments and ARI values
num_segments = [result[0] for result in ari_results]
ari_values = [result[1] for result in ari_results]

# Plotting the results
plt.plot(num_segments, ari_values, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Bootstrapping Clustering Results')
plt.grid(True)
plt.show()


# In[49]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming MD_km28 is a list of KMeans models
# and MD_x is the data matrix

# Extracting the cluster labels for k=4
kmeans_model = MD_km28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)
labels = kmeans_model.labels_

# Plotting the histogram
plt.hist(labels, bins=np.arange(0, 5, 1), edgecolor='black', color='grey')
plt.xlim(0, 4)
plt.xlabel('Cluster Labels')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Labels for k=4')
plt.grid(True)
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

# Assuming MD_km28 is a list of KMeans models 
# and MD_x is the data matrix

# Function to plot silhouette scores for each k
def plot_silhouette(models, data):
    for model in models:
        k = model.n_clusters
        cluster_labels = model.labels_
        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        fig, ax = plt.subplots()
        for i in range(k):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10

        ax.set_title(f"Silhouette plot for k={k}")
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the y-axis labels
        ax.set_xticks(np.arange(-0.1, 1.1, 0.2))

        plt.show()

# Plotting silhouette scores for each model in MD_km28
plot_silhouette(MD_km28, MD_x)



# In[52]:


# Assuming MD_km28 is a list of KMeans models

# Extracting the clustering results for k=4
MD_k4 = MD_km28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)

# Now MD_k4 contains the KMeans model for k=4
print(MD_k4)


# In[53]:


from sklearn.metrics import silhouette_samples, silhouette_score

# Assuming MD_k4 is the KMeans model for k=4 and MD_x is the data matrix

# Calculate silhouette scores for each sample
silhouette_avg = silhouette_score(MD_x, MD_k4.labels_)
sample_silhouette_values = silhouette_samples(MD_x, MD_k4.labels_)

# MD_r4 will store the silhouette scores
MD_r4 = {
    'silhouette_avg': silhouette_avg,
    'sample_silhouette_values': sample_silhouette_values
}

print("Average silhouette score for k=4:", MD_r4['silhouette_avg'])
print("Sample silhouette values for k=4:", MD_r4['sample_silhouette_values'])


# In[54]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming MD_r4 is a dictionary containing silhouette scores as defined in the previous example
# and MD_k4 is the KMeans model for k=4

# Extracting silhouette values
sample_silhouette_values = MD_r4['sample_silhouette_values']

# Plotting the silhouette values
plt.plot(sample_silhouette_values, 'o', color='grey')
plt.ylim(0, 1)
plt.xlabel('Segment Number')
plt.ylabel('Segment Stability')
plt.title('Silhouette Widths for k=4')
plt.grid(True)
plt.show()


# In[55]:


import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming MD_x is your data matrix
np.random.seed(1234)

# Function to fit Gaussian Mixture Models for different values of k
def fit_gmm(data, k_range, nrep=10):
    best_models = []
    for k in k_range:
        best_model = None
        best_bic = np.inf
        for _ in range(nrep):
            model = GaussianMixture(n_components=k, random_state=np.random.randint(0, 10000))
            model.fit(data)
            bic = model.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_model = model
        best_models.append(best_model)
    return best_models

# Fit models for k = 2 to 8
k_range = range(2, 9)
MD_m28 = fit_gmm(MD_x, k_range, nrep=10)

# Print the best models
for k, model in zip(k_range, MD_m28):
    print(f"Best model for k={k}:")
    print(f"Means: {model.means_}")
    print(f"Covariances: {model.covariances_}")
    print(f"BIC: {model.bic(MD_x)}\n")


# In[56]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming MD_x is your data matrix
np.random.seed(1234)

# Function to fit Gaussian Mixture Models for different values of k and calculate AIC, BIC
def fit_gmm_and_calculate_criteria(data, k_range, nrep=10):
    criteria = {'k': [], 'AIC': [], 'BIC': []}
    for k in k_range:
        best_aic = np.inf
        best_bic = np.inf
        for _ in range(nrep):
            model = GaussianMixture(n_components=k, random_state=np.random.randint(0, 10000))
            model.fit(data)
            aic = model.aic(data)
            bic = model.bic(data)
            if aic < best_aic:
                best_aic = aic
            if bic < best_bic:
                best_bic = bic
        criteria['k'].append(k)
        criteria['AIC'].append(best_aic)
        criteria['BIC'].append(best_bic)
    return criteria

# Fit models and calculate criteria for k = 2 to 8
k_range = range(2, 9)
criteria = fit_gmm_and_calculate_criteria(MD_x, k_range, nrep=10)

# Plotting the criteria
plt.plot(criteria['k'], criteria['AIC'], label='AIC', marker='o')
plt.plot(criteria['k'], criteria['BIC'], label='BIC', marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Value of Information Criteria (AIC, BIC)')
plt.title('Information Criteria for Different Numbers of Segments')
plt.legend()
plt.grid(True)
plt.show()



# In[57]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

# Assuming MD_x is your data matrix
np.random.seed(1234)

# Function to fit Gaussian Mixture Models for different values of k and calculate AIC, BIC, ICL
def fit_gmm_and_calculate_criteria(data, k_range, nrep=10):
    criteria = {'k': [], 'AIC': [], 'BIC': [], 'ICL': []}
    for k in k_range:
        best_aic = np.inf
        best_bic = np.inf
        best_icl = np.inf
        for _ in range(nrep):
            model = GaussianMixture(n_components=k, random_state=np.random.randint(0, 10000))
            model.fit(data)
            aic = model.aic(data)
            bic = model.bic(data)
            log_likelihood = model.score(data) * data.shape[0]
            entropy = -np.sum(np.log(model.predict_proba(data)))
            icl = bic + entropy
            if aic < best_aic:
                best_aic = aic
            if bic < best_bic:
                best_bic = bic
            if icl < best_icl:
                best_icl = icl
        criteria['k'].append(k)
        criteria['AIC'].append(best_aic)
        criteria['BIC'].append(best_bic)
        criteria['ICL'].append(best_icl)
    return criteria

# Fit models and calculate criteria for k = 2 to 8
k_range = range(2, 9)
criteria = fit_gmm_and_calculate_criteria(MD_x, k_range, nrep=10)

# Plotting the criteria
plt.plot(criteria['k'], criteria['AIC'], label='AIC', marker='o')
plt.plot(criteria['k'], criteria['BIC'], label='BIC', marker='o')
plt.plot(criteria['k'], criteria['ICL'], label='ICL', marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Value of Information Criteria (AIC, BIC, ICL)')
plt.title('Information Criteria for Different Numbers of Segments')
plt.legend()
plt.grid(True)
plt.show()


# In[59]:


from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Assuming MD_km28 is a list of KMeans models and MD_m28 is a list of GaussianMixture models
# and MD_x is the data matrix

# Extracting the KMeans model for k=4
MD_k4 = MD_km28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)

# Extracting the GaussianMixture model for k=4
MD_m4 = MD_m28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)

# Predicting clusters
kmeans_clusters = MD_k4.predict(MD_x)
mixture_clusters = MD_m4.predict(MD_x)

# Creating a contingency table
contingency_table = pd.crosstab(kmeans_clusters, mixture_clusters, rownames=['kmeans'], colnames=['mixture'])

print(contingency_table)


# In[42]:


from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Assuming MD_km28 is a list of KMeans models and MD_x is the data matrix

# Extracting the KMeans model for k=4
MD_k4 = MD_km28[2]  # Index 2 corresponds to k=4 (since k_range starts from 2)

# Predicting clusters using KMeans
kmeans_clusters = MD_k4.predict(MD_x)

# Fitting a Gaussian Mixture Model with initial cluster assignments from KMeans
MD_m4a = GaussianMixture(n_components=4, random_state=1234)
MD_m4a.fit(MD_x, kmeans_clusters)

# Predicting clusters using the Gaussian Mixture Model
mixture_clusters = MD_m4a.predict(MD_x)

# Creating a contingency table
contingency_table = pd.crosstab(kmeans_clusters, mixture_clusters, rownames=['kmeans'], colnames=['mixture'])

print(contingency_table)


# In[60]:


# Assuming MD_m4a and MD_m4 are GaussianMixture models fitted to the data MD_x

# Calculate log-likelihood for MD.m4a
log_lik_m4a = MD_m4a.score(MD_x) * MD_x.shape[0]
print(f'Log Likelihood for MD.m4a: {log_lik_m4a}')

# Calculate log-likelihood for MD.m4
log_lik_m4 = MD_m4.score(MD_x) * MD_x.shape[0]
print(f'Log Likelihood for MD.m4: {log_lik_m4}')


# In[101]:


#REgression


# In[102]:


df=pd.read_csv('C:\\Users\\Lenovo\\OneDrive\\Desktop\\mcdonalds.csv')


# In[103]:


df['Like'].unique()


# In[104]:


df['Like']


# In[105]:


df['Like'].isnull().sum()


# In[106]:


like_counts = df['Like'].value_counts().sort_index(ascending=False)
like_counts 


# In[109]:


# Mapping categorical levels to numeric codes
mapping = {
    '-3': 1,
    '2': 2,
    '1': 3,
    '4': 4,
    'I love it!+5': 5,
    'I hate it!-5':6,
    '-2': 7,
    '3': 8,
    '0': 9,
    '-4': 10,
    '-1': 11
}

# Replace categorical levels with numeric codes
df['Like'] = df['Like'].replace(mapping)

# Compute the new values
df['Like.n'] = 6 - df['Like']

# Generate the frequency table for 'Like.n'
like_n_counts = df['Like.n'].value_counts().sort_index()
print(like_n_counts)


# In[110]:


df['Like'] = pd.to_numeric(df['Like'])


# In[111]:


df['Like'] = pd.to_numeric(df['Like'])
df['Like.n'] = 6 - df['Like']

# Generate the frequency table for 'Like.n'
like_n_counts = df['Like.n'].value_counts().sort_index()
print(like_n_counts)


# In[112]:


# Step 1: Concatenate the first 11 column names with a '+' separator
f = " + ".join(df.columns[:11])

# Step 2: Create the formula string
formula = f"Like.n ~ {f}"

# Print the formula
print(formula)


# In[113]:


df.columns


# In[118]:


mapping = {
    'Yes': 1,
    'No': 0
} 
df = df.replace(mapping)


# In[121]:


from sklearn.mixture import GaussianMixture
# Set the random seed
np.random.seed(1234)

# Extract the features (replace with your actual feature extraction)
X = df[['ppppppp', 'convenient', 'spicy', 'fattening', 'greasy', 'fast',
       'cheap', 'tasty', 'expensive', 'healthy', 'disgusting',
       'Like.n']]

# Fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, n_init=10, random_state=1234, verbose=0)
gmm.fit(X)

# Print the model summary
print("Converged:", gmm.converged_)
print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)


# In[122]:


gmm_refit = GaussianMixture(n_components=2, n_init=10, random_state=1234, verbose=0)
gmm_refit.fit(X)
# Print the summary of the refitted model
print("Converged:", gmm_refit.converged_)
print("Means:", gmm_refit.means_)
print("Covariances:", gmm_refit.covariances_)
print("Weights:", gmm_refit.weights_)


# In[ ]:




