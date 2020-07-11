#!/bin/bash

files=./weather/unzipped
encoded="_encoded"
zip='zip'
#cd weather
#mkdir unzipped
#mkdir encoded
echo $PWD

#7za x "*.zip" -ounzipped
cd weather/unzipped
echo $PWD
for file in *.csv
do
	iconv -f iso-8859-1 -t utf-8 "$file" > "${file%.csv}_encoded.csv"
done

