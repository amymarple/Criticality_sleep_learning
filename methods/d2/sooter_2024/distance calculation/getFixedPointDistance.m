function [distance, closest_model] = getFixedPointDistance(order, beta, model)
%beta designates which basin of attraction to calculate the distance to
%model is the history kernel of the AR model

m = beta/2-1;
%construct points
x = zeros(order, order-m); %each column is one point (one history kernel)
for k = (m+1):order
    for t = 1:k
        x(t, k-m) = nchoosek(k, t)*(-1)^(t+1);
    end

    for t = (k+1):order
        x(t, k-m) = 0;
    end
end

X = x - x(:, 1);
X = X(:, 2:end);
if size(model, 1)<size(model, 2)
    model = model';
end

bhat = model - x(:, 1);

%QR decomposition
[Q,R] = qr(X);
v_qr = R \ (Q'*bhat);

closest_model = x(:, 1) + X*v_qr;
distance = sqrt(sum((model-closest_model).^2));
end