w11=0.1;
w12=0.25;
w21=0.1;
w22=0.7;
w13=0.4;
w14=0.5;
w23=0.6;
w24=0.3;
v1=w11*x1+w21*x2;
v2=w12*x1+w22*x2;
y1=1/(1+exp(-1*v1));
y2=1/(1+exp(-1*v2));
v3=w13*y1+w23*y2;
v4=w14*y1+w24*y2;
y3=1/(1+exp(-1*v3));
y4=1/(1+exp(-1*v4));
d=[1;0];
pred=[y3;y4];
E=sum((pred-d).^2/2);
Edy3=y3-1;
Edy4=y4-0;
y3dv3=y3^2*exp(-v3);
y4dv4=y4^2*exp(-v4);
v3dw13=y1;
v3dw23=y2;
v4dw14=y1;
v4dw24=y2;
Edw13=Edy3*y3dv3*v3dw13;
Edw14=Edy4*y4dv4*v4dw14;
Edw23=Edy3*y3dv3*v3dw23;
Edw24=Edy4*y4dv4*v4dw24;

nw13=w13-0.1*Edw13;
nw14=w14-0.1*Edw14;
nw23=w23-0.1*Edw23;
nw24=w24-0.1*Edw24;


v3dy1=w13;
v3dy2=w23;
v4dy1=w14;
v4dy2=w24;

Ey3dv3=Edy3*y3dv3;
Ey4dv4=Edy4*y4dv4;

Ey3dy1=Ey3dv3*v3dy1;
Ey4dy1=Ey4dv4*v4dy1;
Ey3dy2=Ey3dv3*v3dy2;
Ey4dy2=Ey4dv4*v4dy2;

Edy1=Ey3dy1+Ey4dy1;
Edy2=Ey3dy2+Ey4dy2;

y1dv1=y1^2*exp(-v1);
y2dv2=y2^2*exp(-v2);

v1dw11=x1;
v2dw12=x1;
v1dw21=x2;
v2dw22=x2;



Edw11=Edy1*y1dv1*v1dw11;
Edw12=Edy2*y2dv2*v2dw12;
Edw21=Edy1*y1dv1*v1dw21;
Edw22=Edy2*y2dv2*v2dw22;

nw11=w11-0.1*Edw11;
nw12=w12-0.1*Edw12;
nw21=w21-0.1*Edw21;
nw22=w22-0.1*Edw22;
