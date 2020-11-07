import os 
  
# Function to rename multiple files 
def main(): 
  for count, filename in enumerate(os.listdir("./")):
    if 'tesabob' in filename:
      os.rename('./' + filename, filename[21:])

if __name__ == '__main__': 
  main()