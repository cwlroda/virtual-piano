import os 
  
# Function to rename multiple files 
def main(): 
  for count, filename in enumerate(os.listdir("./")):
    if 'tesabob' in filename:
      print(filename[21:])
      os.rename('./' + filename, filename[21:])
    # dst ="Hostel" + str(count) + ".jpg"
    # src ='xyz'+ filename 
    # dst ='xyz'+ dst 
      
    # # rename() function will 
    # # rename all the files 
    # os.rename(src, dst) 

if __name__ == '__main__': 
  main()