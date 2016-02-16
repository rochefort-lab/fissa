macro "Oval2PolyROI [G]" {
    ROIpath = getInfo("image.directory");
    ROIfiletif = getTitle();
    dotIndex = indexOf(ROIfiletif, "."); 
    ROIfile = substring(ROIfiletif, 0,dotIndex);
    roiManager("open", ROIpath+"\\"+ROIfile+".zip");

    n_ROIs = roiManager("count")
    for (i=0;i<n_ROIs;i++){
	roiManager("Select", i);
	getSelectionCoordinates(x, y); 
	makeSelection("plygon", x, y);
	roiManager("Add")
	roiManager("Update") 	
	}

    for (i=0;i<n_ROIs;i++){
       	roiManager("Select", 0);
        roiManager("Delete")
        }
        
    ROIcnt = roiManager("count");
    imgdir = getInfo("image.directory");
    filenametif = getTitle();
    dotIndex = indexOf(filenametif, "."); 
    filename = substring(filenametif, 0,dotIndex)
    i = 1;
    while (i<(ROIcnt+1)) {
      roiManager("select", (i-1))
	 if (i<10) {
       		roiManager("Rename", "ROI_00" + i);
         } else  
         if (i<100) {
       		roiManager("Rename", "ROI_0" + i);
	 } else  {
       		roiManager("Rename", "ROI_" + i);
	 }
	 roiManager("Save", imgdir+"\\"+filename+".zip"); 
	 i = i+1;
  
  }
run("Close");
}