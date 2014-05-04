f=avgScores.txt
for var in 1 2 3 4 5 6 7 8 9 10
do
  printf "%3d " "$var" >> $f
  python pacman.py -T RLated -p HardCodedAgent -m 100 -s $var -q -n 50 | grep "Average" >> $f
done
 
