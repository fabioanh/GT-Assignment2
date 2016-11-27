#
PREFIX="./outputs"   # define where to store all the outputs
mkdir -p "${PREFIX}"    # make sure the outputs dir exists

for FILE in ./8_moo*/summary.out       # get the file names you want to work on
do
  # use ${PREFIX}/${FILE} to redirect output to a 
  # file that's associated with the input
  echo 
  grep 'Coop Lvl History:' "${FILE}"  >> "${PREFIX}/8_moore_coop_lvl_history.csv"
done