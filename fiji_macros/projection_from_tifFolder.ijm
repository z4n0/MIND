// Maximum Intensity Projection for .tif files in a folder

// Prompt user for input and output folders
input  = getDirectory("Select folder containing .tif files");
output = getDirectory("Select folder to save projected .tif images");

// Get list of files in the input folder
tif_list = getFileList(input);

setBatchMode(true);

// Process each .tif file in the folder
for (i = 0; i < tif_list.length; i++) {
    fileName = tif_list[i];
    
    if (endsWith(fileName, ".tif") || endsWith(fileName, ".TIF")) {
        fullPath = input + fileName;
        print("Processing TIF:", fullPath);
        
        // Open the .tif file
        open(fullPath);
        
        // Perform Maximum Intensity Projection
        run("Z Project...", "projection=[Max Intensity]");
        
        // Save the projection
        projName = getTitle();
        outPath = output + projName;
        saveAs("Tiff", outPath);
        print("  Saved ->", outPath);
        
        // Close the projection and original image
        close();
        if (nImages() > 0) {
            close();
        }
    }
}

setBatchMode(false);
print("Done!");
