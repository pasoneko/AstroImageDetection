conda activate new_hal

MAP="/lustre/hawcz01/hawcroot/maps/pass-5.final/Pass5-Final-2pctfhit-Map-ch1-1510/chunks1-1510_000001-001510_N1024_sig.fits.gz"

read -p "ROI-center coord1: " CENTER1

read -p "ROI-center coord2: " CENTER2

#read -p "size 1" SIZE1

#read -p "size 2" SIZE2

read -p "output directory: " OUTDIR

SIZES=(10 8 6 4 2 1)

for SIZE in "${SIZES[@]}"; do
    OUTDIRX="$OUTDIR/$SIZE"

    python finder.py -M $MAP --ROI-center $CENTER1 $CENTER2 --size $SIZE $SIZE -O $OUTDIRX
done

echo "Complete"