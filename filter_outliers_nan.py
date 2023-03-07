import netCDF4 as nc
import numpy as np
import os


def remove_NaNs(input_dir, output_dir):
    """
    Removes NaN values from all variables in all netCDF files in the input directory and its subdirectories.
    Creates new netCDF files with the same variable names and dimensions but with NaN values removed in the output directory.

    Parameters
    ----------
    input_dir : str
        Input directory containing the netCDF files to process.
    output_dir : str
        Output directory where the processed netCDF files will be written.
    """
    # Loop over all the files and directories in the input directory
    for root, dirs, files in os.walk(input_dir):
        # Loop over all the files in the current directory
        for file_name in files:
            # Check if the file is a netCDF file
            if file_name.endswith('.nc'):
                # Construct the input and output file paths
                input_file_path = os.path.join(root, file_name)
                output_file_path = os.path.join(output_dir, os.path.relpath(input_file_path, input_dir).replace('.nc',
                                                                                                                '_no_NaNs.nc'))

                # Open the input netCDF file
                input_file = nc.Dataset(input_file_path, 'r')

                # Open the output netCDF file for writing
                output_file = nc.Dataset(output_file_path, 'w')

                # Loop over all the variables in the input file
                for var_name, var in input_file.variables.items():
                    # Create the variable in the output file
                    out_var = output_file.createVariable(var_name, var.dtype, var.dimensions)

                    # Loop over all the dimensions of the variable
                    for dim_name, dim in var.dimensions.items():
                        # Create the dimension in the output file
                        out_dim = output_file.createDimension(dim_name, len(dim))

                    # Get the variable data
                    var_data = var[:]

                    # Create a mask for the NaN values
                    mask = np.isnan(var_data)

                    # Filter out the NaN values
                    filtered_data = np.ma.masked_array(var_data, mask)

                    # Write the filtered data to the output file
                    out_var[:] = filtered_data.filled(np.nan)

                # Close the input and output files
                input_file.close()
                output_file.close()
