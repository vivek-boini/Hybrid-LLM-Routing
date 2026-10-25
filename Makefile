.PHONY: test dryrun clean

test:
	python -m pytest tests -v

dryrun:
	python -m qos.cli synthetic --out data/synthetic.npz
	python -m qos.cli predict --data data/synthetic.npz --epochs 5
	python -m qos.cli eval --data data/synthetic.npz --lam 5 --n-req 100 --policies rr,sqf,br,greedy

clean:
	rm -rf __pycache__ qos/__pycache__ tests/__pycache__ .pytest_cache
