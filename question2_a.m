%load data
x = load('ex4x.dat');
y = load('ex4y.dat');


% Add x0=1 intercpt term into x matrix
x = [ones(size(x,1), 1), x]; 

%plot the data
pos = find(y == 1); neg = find(y == 0);

plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o');hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')

%sigmoid function
g = inline('1.0 ./ (1.0 + exp(-z))'); 

%Newton's method
theta = zeros(size(x,2), 1);

iteration = 10;
J = zeros(iteration, 1);
for i=1: iteration
    
    z = x * theta;
    h = g(z);
    
    m =size(x,1);
    gradient = (1/m)*(x'*(h-y));
    hessian = (1/m)*(x'*diag(h)*diag(1-h)*x);
    
    theta = theta - hessian\gradient;
    
    J(i) =(1/m)*sum(-y.*log(h) - (1-y).*log(1-h));
end

theta
J

% Plot Newton's method result
plot_x = [min(x(:,2)),  max(x(:,2))];
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)

%calculate the probability
xx = [1, 30, 70];
z = xx*theta;
prob = 1 - g(z)
