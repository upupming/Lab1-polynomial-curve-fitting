all: least_squares least_squares_regularization gradient_descent conjugate_gradient

clean:
	rm -rf ./training_results ./images

create_dirs:
	mkdir -p training_results images

least_squares: create_dirs
	python ./src/least_squares.py
least_squares_regularization: create_dirs
	python ./src/least_squares_regularization.py
gradient_descent: create_dirs
	python ./src/gradient_descent.py
conjugate_gradient: create_dirs
	python ./src/conjugate_gradient.py