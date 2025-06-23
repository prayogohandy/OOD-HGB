import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def print_ranges(df):
    ranges = pd.DataFrame({
        'Min': df.min(),
        'Max': df.max()
    })
    print(ranges)


def process_dataframe(df, scaler = 'standard'):
    """
    Process a dataframe by separating numerical and categorical columns based on their data types, and scaling numerical columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to be processed, containing both numerical and categorical columns.
    scaler : str, optional
        The scaling method to apply to numerical columns. Options are:
        - 'none' : No scaling is applied.
        - 'standard' : Applies StandardScaler (mean=0, variance=1).
        - 'minmax' : Applies MinMaxScaler (scales values between 0 and 1).

    Returns:
    --------
    pd.DataFrame
        The processed dataframe with scaled numerical columns and one-hot encoded categorical columns.
    int
        The length of numerical columns.
    list
        List of the length of OHE columns.
    dict
        A dictionary containing column names, the fitted scaler object, and the list of fitted OneHotEncoder objects.
    """
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Initialize a list to store individual OHE DataFrames
    ohe_dfs = []
    
    # Initialize a dictionary to store OHE column indices for each categorical feature
    len_ohes = []
    ohe_scalers = []
    col_index = len(numerical_cols)  # Start counting after numerical columns

    # Initialize the scaler based on the argument
    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'none': None
    }
    
    if scaler not in scaler_dict:
        raise ValueError(f"Scaler '{scaler}' not recognized. Choose from 'standard', 'minmax', or 'none'.")
    
    selected_scaler = scaler_dict[scaler]
    
    # Scale numerical columns
    if selected_scaler is not None:
        df[numerical_cols] = selected_scaler.fit_transform(df[numerical_cols])
     
    # Process each categorical column separately
    for col in categorical_cols:
        # Apply OHE to the column
        ohe = OneHotEncoder()
        ohe_transformed = ohe.fit_transform(df[[col]]).toarray()  # df[[col]] to keep it as a DataFrame
        
        # Get the new column names after OHE
        ohe_columns = ohe.get_feature_names_out([col])
        
        # Create a DataFrame from OHE transformation
        ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_columns, index=df.index)
        
        # Append the OHE DataFrame to the list
        ohe_dfs.append(ohe_df)
        
        # Store the fitted OneHotEncoder for reversing later
        ohe_scalers.append(ohe)
        
        # Store the starting index of the first OHE column for this feature
        len_ohes.append(len(ohe_df.columns))
        
        # Update the starting index for the next feature's OHE columns
        col_index += len(ohe_df.columns)
    
    # Concatenate all OHE DataFrames horizontally (axis=1)
    final_ohe_df = pd.concat(ohe_dfs, axis=1)

    # Combine numerical columns and all OHE DataFrames
    final_df = pd.concat([df[numerical_cols], final_ohe_df], axis=1)

    # Get column indices for numerical columns
    len_numerical = len(numerical_cols)

     # Create a dictionary to store additional info
    additional_info = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'selected_scaler': selected_scaler,
        'ohe_scalers': ohe_scalers
    }

    # Return the processed DataFrame and additional info
    return final_df, len_numerical, len_ohes, additional_info


def reverse_process_dataframe(processed_data, len_numerical, len_ohes, additional_info):
    """
    Reverse the processed dataframe or tensor array to its original form by reversing scaling on numerical columns
    and converting one-hot encoded columns back to categorical values using the fitted OHE scalers.

    Parameters:
    -----------
    processed_data : pd.DataFrame or torch.Tensor
        The processed data with scaled numerical columns and one-hot encoded categorical columns.
    len_numerical : int
        The number of numerical columns in the original dataframe.
    len_ohes : list
        List containing the number of one-hot encoded columns for each categorical feature.
    additional_info : dict
        A dictionary containing column names, the scaler that was used during processing, 
        and the list of fitted OneHotEncoder objects.

    Returns:
    --------
    pd.DataFrame
        The reversed dataframe with original numerical and categorical columns.
    """

    scaler = additional_info['selected_scaler']
    ohe_scalers = additional_info['ohe_scalers']
    numerical_cols = additional_info['numerical_cols']
    categorical_cols = additional_info['categorical_cols']

    # Convert to NumPy array if processed_data is a PyTorch tensor
    if isinstance(processed_data, torch.Tensor):
        processed_data = processed_data.detach().cpu().numpy()
    else:
        processed_data = processed_data.to_numpy()

    # Step 1: Reverse scaling for numerical columns
    if scaler is not None:
        numerical_data = scaler.inverse_transform(processed_data[:, :len_numerical])
    else:
        numerical_data = processed_data[:, :len_numerical]

    # Step 2: Reverse one-hot encoding for categorical columns
    start_idx = len_numerical
    categorical_data = {}

    for i, (ohe_len, ohe_scaler, cat_col) in enumerate(zip(len_ohes, ohe_scalers, categorical_cols)):
        # Extract the one-hot encoded columns for this categorical feature
        ohe_columns = processed_data[:, start_idx:start_idx + ohe_len]

        # Reverse the one-hot encoding using the fitted OHE scaler
        categorical_data[cat_col] = ohe_scaler.inverse_transform(ohe_columns)

        # Move to the next set of OHE columns
        start_idx += ohe_len

    # Step 3: Combine numerical and categorical columns
    reversed_df = pd.DataFrame(numerical_data, columns=numerical_cols)

    for cat_col, cat_data in categorical_data.items():
        reversed_df[cat_col] = cat_data.flatten()  # Flatten to get rid of extra dimension

    return reversed_df

def numpy_to_tensor(numpy_dict, dtype = None):
    """
    Converts a dictionary of NumPy arrays to a dictionary of PyTorch tensors.

    Args:
        numpy_dict (dict): A dictionary where the values are NumPy arrays.
        dtype (tensor.

    Returns:
        dict: A dictionary where the values are PyTorch tensors.
    """
    if dtype is None:
        return {key: (torch.tensor(value) if not isinstance(value, torch.Tensor) else value)
                for key, value in numpy_dict.items()}
    else:
        return {key: (torch.tensor(value, dtype=dtype) if not isinstance(value, torch.Tensor) else value)
                for key, value in numpy_dict.items()}

def tensor_to_numpy(tensor_dict):
    """
    Converts a dictionary of PyTorch tensors to a dictionary of NumPy arrays.

    Args:
        tensor_dict (dict): A dictionary where the values are PyTorch tensors.

    Returns:
        dict: A dictionary where the values are NumPy arrays.
    """
    return {key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in tensor_dict.items()}

def generate_nd_gaussian(a, b, d=3, num_samples=50):
    """
    Generate d-dimensional Gaussian samples from a standard normal distribution N(0,1) with a specified radius range [a, b].

    Parameters:
    -----------
    a : float
        The minimum radius of the generated samples.
    b : float
        The maximum radius of the generated samples.
    d : int
        The number of dimensions for each sample.
    num_samples : int
        The number of samples to generate.

    Returns:
    --------
    np.ndarray
        A numpy array of shape (num_samples, d), containing d-dimensional Gaussian samples constrained to have radii 
        within the range [a, b].
    """
    # Compute the CDF values at the limits a^2 and b^2 for the chi-squared distribution with d degrees of freedom
    lower_bound = chi2.cdf(a**2, df=d) 
    upper_bound = chi2.cdf(b**2, df=d)
    
    # Sample uniformly between the CDF values of the bounds
    uniform_samples = np.random.uniform(lower_bound, upper_bound, size=num_samples)
    
    # Invert the CDF to get the corresponding chi-squared values
    r_squared_samples = chi2.ppf(uniform_samples, df=d)
    
    # Take the square root to get the radius
    radii = np.sqrt(r_squared_samples)
    
    # Generate random points on the d-dimensional unit hypersphere
    directions = np.random.normal(size=(num_samples, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize to get unit vectors
    
    # Scale the unit directions by the sampled radii
    samples = directions * radii[:, np.newaxis]
    
    return samples


def generate_nd_ball_sample(num_points, dimension, radius):
    """
    Generate random points uniformly distributed within a d-dimensional sphere of a given radius.

    Parameters:
    num_points : int
        Number of points to generate.
    dimension : int
        Dimensionality of the sphere.
    radius : float
        Radius of the sphere.

    Returns:
    np.ndarray
        Array of shape (num_points, dimension) containing the random points.
    """
    # Generate random directions by normalizing the length of a vector of random-normal values
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)

    # Generate random radii with probability proportional to the surface area of a ball
    random_radii = np.random.random(num_points) ** (1 / dimension)

    # Return the points in the d-dimensional sphere
    return radius * (random_directions * random_radii).T

def l2dist(samples):
    """
    Calculate the L2 (Euclidean) distance of the input samples along axis 1.

    Parameters:
    -----------
    samples : np.ndarray
        A 2D array where each row represents a sample and each column represents a feature.

    Returns:
    --------
    np.ndarray
        A 1D array of L2 distances for each sample, calculated along axis 1.
    """
    return np.linalg.norm(samples, axis=1, keepdims=True)

def plot_embed(embedding, color_labels=None, x_label='Latent Variable 1', y_label='Latent Variable 2', 
               z_label='Latent Variable 3', point_size=4, axis_limits=None, filename=None):
    """
    Visualize high-dimensional embeddings in either 2D or 3D space using Plotly.

    If the embedding has 3 dimensions, a 3D scatter plot will be created. If it has 2 dimensions,
    a 2D scatter plot will be created. Optionally, labels for coloring the points can be provided.

    Parameters:
    -----------
    embedding : np.ndarray
        A 2D array where each row represents a sample's embedding. The number of columns must be 2 or 3.

    color_labels : array-like, optional
        A 1D array of labels corresponding to the samples in `embedding`. The length should match
        the number of rows in `embedding`. If provided, these labels will be used for coloring the points.

    x_label : str, optional
        Label for the X-axis. Default is 'Latent Variable 1'.

    y_label : str, optional
        Label for the Y-axis. Default is 'Latent Variable 2'.

    z_label : str, optional
        Label for the Z-axis. Default is 'Latent Variable 3'. Only used for 3D plots.

    point_size : int, optional
        Size of the scatter points. Default is 2.

    axis_limits : dict, optional
        Dictionary with axis limits. Keys should be 'x', 'y', and 'z' for the respective axis limits as tuples (min, max).

    Returns:
    --------
    None
        The function directly displays the plot using Plotly.
    """
    if embedding.shape[1] not in [2, 3]:
        raise ValueError("Embedding must have exactly 2 or 3 dimensions.")

    if color_labels is not None and len(color_labels) != embedding.shape[0]:
        raise ValueError("Length of color_labels must match the number of rows in embedding.")

    if embedding.shape[1] == 3:
        # 3D Scatter Plot
        fig = px.scatter_3d(
            embedding, 
            x=0, y=1, z=2, 
            color=color_labels,
            size_max=point_size
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            autosize=False,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins as needed
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.75, y=1.5, z=1.5))
            )
        )
        # Set axis limits if provided
        if axis_limits is not None:
            if 'x' in axis_limits:
                fig.update_layout(scene=dict(xaxis=dict(range=axis_limits['x'])))
            if 'y' in axis_limits:
                fig.update_layout(scene=dict(yaxis=dict(range=axis_limits['y'])))
            if 'z' in axis_limits:
                fig.update_layout(scene=dict(zaxis=dict(range=axis_limits['z'])))
    else:
        # 2D Scatter Plot
        fig = px.scatter(
            embedding, 
            x=0, y=1, 
            color=color_labels,
            size_max=point_size
        )
        fig.update_traces(marker=dict(size=point_size))
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins as needed
        )
        # Set x-axis and y-axis limits if provided
        if axis_limits is not None:
            if 'x' in axis_limits:
                fig.update_layout(xaxis=dict(range=axis_limits['x']))
            if 'y' in axis_limits:
                fig.update_layout(yaxis=dict(range=axis_limits['y']))
    fig.show()

    if filename is not None:
        pio.write_image(fig, filename, scale = 3)
    
def distribution_comparison(real_data, synthetic_data, n_bins=10, columns=None, stacked=False, 
                            rename_dict=None, filename=None):
    """
    Compare distributions of real and synthetic data with customizable column names and an option to save the plot.

    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with real data.

    synthetic_data : pd.DataFrame
        DataFrame with synthetic data.

    n_bins : int, optional
        Number of bins for histograms. Default is 10.

    columns : list of str, optional
        Specific columns to plot. If None, all columns are plotted.

    stacked : bool, optional
        If True, plots each column in a single subplot. If False, uses side-by-side subplots for real and synthetic.

    rename_dict : dict, optional
        Dictionary to rename columns for display.

    filename : str, optional
        Path to save the plot. If None, the plot will only be displayed.

    Returns:
    --------
    None
        The function displays and optionally saves the plot.
    """
    if columns is None:
        columns = real_data.columns  # Use all columns if not specified

    num_columns = len(columns)
    if stacked:
        fig, axes = plt.subplots(num_columns, 1, figsize=(5, 5 * num_columns))
    else:
        fig, axes = plt.subplots(num_columns, 2, figsize=(10, 5 * num_columns))

    for i, col in enumerate(columns):
        real_col = real_data[col]
        synthetic_col = synthetic_data[col]

        # Determine x-axis label from rename_dict or default to column name
        x_label = rename_dict[col] if rename_dict and col in rename_dict else col

        # Check if the column is categorical
        if real_col.dtype == 'object' or synthetic_col.dtype == 'object':

            # Map real_col and synthetic_col if categorical and rename_dict is provided
            if rename_dict:
                real_col = real_col.map(rename_dict)
                synthetic_col = synthetic_col.map(rename_dict)
            
            ax = axes[i] if stacked else axes[i, 0]
            sns.histplot(real_col, ax=ax, color='blue', label='Real', discrete=True)
            ax.set_xlabel(x_label)
            
            ax = axes[i] if stacked else axes[i, 1]
            sns.histplot(synthetic_col, ax=ax, color='orange', label='Synthetic', discrete=True)
            ax.set_xlabel(x_label)
            ax.legend()
        else:
            # Determine the common bin range
            min_val = min(real_col.min(), synthetic_col.min())
            max_val = max(real_col.max(), synthetic_col.max())
            bin_range = np.linspace(min_val, max_val, num=n_bins)

            if stacked:
                ax = axes[i]
                sns.histplot(real_col, kde=True, ax=ax, color='blue', label='Real', alpha=0.5, bins=bin_range)
                sns.histplot(synthetic_col, kde=True, ax=ax, color='orange', label='Synthetic', alpha=0.5, bins=bin_range)
                ax.set_xlabel(x_label)
                ax.legend()
            else:
                # Plot real data distribution
                sns.histplot(real_col, kde=True, ax=axes[i, 0], color='blue', label='Real', bins=bin_range)
                axes[i, 0].set_title(f"Real Data Distribution: {x_label}")
                axes[i, 0].set_xlabel(x_label)

                # Plot synthetic data distribution
                sns.histplot(synthetic_col, kde=True, ax=axes[i, 1], color='orange', label='Synthetic', bins=bin_range)
                axes[i, 1].set_title(f"Synthetic Data Distribution: {x_label}")
                axes[i, 1].set_xlabel(x_label)

                # Set y-axis limits to be the same for side-by-side comparison
                max_y = max(axes[i, 0].get_ylim()[1], axes[i, 1].get_ylim()[1])
                axes[i, 0].set_ylim(0, max_y)
                axes[i, 1].set_ylim(0, max_y)

    plt.tight_layout()

    # Save the plot if filename is provided
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=600)
    
    plt.show()
    
def corrmap_comparison(real_data, synthetic_data, rename_dict=None, filename_prefix=None):
    """
    Compare correlation matrices of real and synthetic data with optional renaming and saving options.

    Parameters:
    -----------
    real_data : pd.DataFrame
        DataFrame with real data.

    synthetic_data : pd.DataFrame
        DataFrame with synthetic data.

    rename_dict : dict, optional
        Dictionary to rename columns for display.

    filename_prefix : str, optional
        Prefix to use when saving the correlation matrix plots. If None, plots are only displayed.

    Returns:
    --------
    None
        Displays and optionally saves the correlation matrix plots.
    """
    # Function to perform One-Hot Encoding on categorical columns
    def one_hot_encode(df):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(df, columns=categorical_cols)

    # Apply One-Hot Encoding
    real_data_encoded = one_hot_encode(real_data)
    synthetic_data_encoded = one_hot_encode(synthetic_data)

    # Calculate correlation matrices
    real_corr = real_data_encoded.corr()
    synthetic_corr = synthetic_data_encoded.corr()

    # Define a common order for the heatmap (using real_corr as reference)
    common_order = real_corr.columns

    # Reindex both correlation matrices to match this common order
    real_corr = real_corr.reindex(index=common_order, columns=common_order)
    synthetic_corr = synthetic_corr.reindex(index=common_order, columns=common_order)
    
    # Rename columns and index if rename_dict is provided
    if rename_dict:
        real_corr = real_corr.rename(index=rename_dict, columns=rename_dict)
        synthetic_corr = synthetic_corr.rename(index=rename_dict, columns=rename_dict)

    # Set up the figure and axes for correlation maps
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Plot the correlation matrix for real data
    sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    axes[0].set_title('Real Data Correlation Matrix')

    # Plot the correlation matrix for synthetic data
    sns.heatmap(synthetic_corr, ax=axes[1], cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    axes[1].set_title('Synthetic Data Correlation Matrix')

    # Save the correlation matrices if filename_prefix is provided
    if filename_prefix:
        fig.savefig(f'{filename_prefix}_correlation_matrices.png', bbox_inches='tight', dpi=600)

    # Plot the difference in correlation matrices
    fig_diff, ax_diff = plt.subplots(figsize=(10, 10))
    diff_corr = abs(real_corr - synthetic_corr)
    sns.heatmap(diff_corr, ax=ax_diff, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
    ax_diff.set_title('Difference in Correlation (Real - Synthetic)')
    ax_diff.set_xlabel('Features')
    ax_diff.set_ylabel('Features')

    # Save the difference matrix if filename_prefix is provided
    if filename_prefix:
        fig_diff.savefig(f'{filename_prefix}_correlation_difference.png', bbox_inches='tight', dpi = 600)

    plt.tight_layout()
    plt.show()
    
def filter_samples(X, y, ood_label=0):
    """Filter ID and OOD samples based on the specified ID label."""
    is_id = ~(y == ood_label)  # Create a boolean mask for ID labels
    X_id = X[is_id]          # Select ID samples
    y_id = y[is_id]
    
    X_ood = X[~is_id]        # Select OOD samples (reverse mask)
    y_ood = y[~is_id]

    return (X_id, y_id), (X_ood, y_ood)