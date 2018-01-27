topfolder=../02_inputdata
inputfolder=01_colour
outputfolder=02_monochrome

for folder in 01_a 02_i 03_u 04_e 05_o
do
  for filename in `ls $topfolder/$folder/$inputfolder`
  do
    echo $filename
    inputfilepath=$topfolder/$folder/$inputfolder/$filename
    outputfiilepath=$topfolder/$folder/$outputfolder/$filename
    echo "input  : $inputfilepath"
    echo "output : $outputfiilepath"
    convert $inputfilepath -type Grayscale $outputfiilepath
  done
done
