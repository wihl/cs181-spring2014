grep -c $1 * | awk -F : '{total += $2} END { print "Total:", total }'
