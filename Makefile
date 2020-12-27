PY := python3
PROG := analyze.py

# Run data analysis script.
release : $(PROG)
	$(PY) $(PROG)

# Remove executable binary and generated objected files.
.PHONY : clean
clean : 
	rm -f *.png matching.csv pearson_correlation.csv percent_diff.csv
