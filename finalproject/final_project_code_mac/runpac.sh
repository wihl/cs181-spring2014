f=avgScores.txt
for var in {1..10}
do
  printf "%3d " "$var" >> $f
  python pacman.py -T RLated -p HardCodedAgent -m 100 -s $var -q -n 50 | grep "Average" | sed "s/Average Score://" >> $f
done
 
