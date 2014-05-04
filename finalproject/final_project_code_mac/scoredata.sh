f=avgScores.txt
for var in {1..20}
do
  printf "%3d " "$var" >> $f
  python pacman.py -T RLated -p HardCodedAgent -m 200 -s $var -q -n 50 | grep "Scores" >> $f
done
 
