
def FileSelect(FileDescriptors, Extensions, plural):

    root = tk.Tk()                     
    root.withdraw()
    root.attributes("-topmost", True)
    root.focus_force()

    if plural == True:
        PathReturn = fd.askopenfilenames(filetypes=[(FileDescriptors, Extensions)])

    elif plural == False:
        PathReturn = fd.askopenfilename(filetypes=[(FileDescriptors, Extensions)])
    
    root.destroy()
    return PathReturn



print('Well Data Cutter Tool')

import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd

Well_File_Path = FileSelect('Well Plate File', '.csv', False)                   #Function that lets user select file with file picker and return the file path into the variable

Well_File_df = pd.read_csv(Well_File_Path)                                      #Load current CSV file into panda dataframe 

A1_df = Well_File_df.copy()
#A1_df = A1_df.drop(A1_df.filter(like='A1').columns, axis=1)                    <---- Skips deleting this becuase its commented out, (this is the one we want to keep)
A1_df = A1_df.drop(A1_df.filter(like='A2').columns, axis=1)                     # Delete all columns with A2 in the name 
A1_df = A1_df.drop(A1_df.filter(like='A3').columns, axis=1)                     # Deletes all columns with A3...
A1_df = A1_df.drop(A1_df.filter(like='B1').columns, axis=1)                     # ...
A1_df = A1_df.drop(A1_df.filter(like='B2').columns, axis=1)
A1_df = A1_df.drop(A1_df.filter(like='B3').columns, axis=1)
A1_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\A1.csv', index=False)                            # Takes panda data frame and writes it to a CSV file Set index=False to avoid writing row numbers



A2_df = Well_File_df.copy()                                                     #Do the same this but now for A2,,,
A2_df = A2_df.drop(A2_df.filter(like='A1').columns, axis=1)
#A2_df = A2_df.drop(A2_df.filter(like='A2').columns, axis=1)
A2_df = A2_df.drop(A2_df.filter(like='A3').columns, axis=1)
A2_df = A2_df.drop(A2_df.filter(like='B1').columns, axis=1)
A2_df = A2_df.drop(A2_df.filter(like='B2').columns, axis=1)
A2_df = A2_df.drop(A2_df.filter(like='B3').columns, axis=1)
A2_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\A2.csv', index=False)  


A3_df = Well_File_df.copy()
A3_df = A3_df.drop(A3_df.filter(like='A1').columns, axis=1)
A3_df = A3_df.drop(A3_df.filter(like='A2').columns, axis=1)
#A3_df = A3_df.drop(A3_df.filter(like='A3').columns, axis=1)
A3_df = A3_df.drop(A3_df.filter(like='B1').columns, axis=1)
A3_df = A3_df.drop(A3_df.filter(like='B2').columns, axis=1)
A3_df = A3_df.drop(A3_df.filter(like='B3').columns, axis=1)
A3_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\A3.csv', index=False)  


B1_df = Well_File_df.copy()
B1_df = B1_df.drop(B1_df.filter(like='A1').columns, axis=1)
B1_df = B1_df.drop(B1_df.filter(like='A2').columns, axis=1)
B1_df = B1_df.drop(B1_df.filter(like='A3').columns, axis=1)
#B1_df = B1_df.drop(B1_df.filter(like='B1').columns, axis=1)
B1_df = B1_df.drop(B1_df.filter(like='B2').columns, axis=1)
B1_df = B1_df.drop(B1_df.filter(like='B3').columns, axis=1)
B1_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\B1.csv', index=False)  


B2_df = Well_File_df.copy()
B2_df = B2_df.drop(B2_df.filter(like='A1').columns, axis=1)
B2_df = B2_df.drop(B2_df.filter(like='A2').columns, axis=1)
B2_df = B2_df.drop(B2_df.filter(like='A3').columns, axis=1)
B2_df = B2_df.drop(B2_df.filter(like='B1').columns, axis=1)
#B2_df = B2_df.drop(B2_df.filter(like='B2').columns, axis=1)
B2_df = B2_df.drop(B2_df.filter(like='B3').columns, axis=1)
B2_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\B2.csv', index=False)  


B3_df = Well_File_df.copy()
B3_df = B3_df.drop(B3_df.filter(like='A1').columns, axis=1)
B3_df = B3_df.drop(B3_df.filter(like='A2').columns, axis=1)
B3_df = B3_df.drop(B3_df.filter(like='A3').columns, axis=1)
B3_df = B3_df.drop(B3_df.filter(like='B1').columns, axis=1)
B3_df = B3_df.drop(B3_df.filter(like='B2').columns, axis=1)
#B3_df = B3_df.drop(B3_df.filter(like='B3').columns, axis=1)
B3_df.to_csv(r'C:\Users\Lindsay\Github\MEA_data\Well_Cutter_Tool\Cut_Well_Plates\B3.csv', index=False)  





print('Done! :) Files are located in the "Cut_Well_Plates folder"')
print('Luv U!')