clear
clc
% x1=w1, x2=w2, x3=y, x4=s, x5 = d, F= f_m=5, U=U_m =5, C=C_o = 3GHz,
% V=5MB, R = R_m = 20 Mbps,  m=c_m^l=2, P_m = 5W ,omaga = 0.5
  
f='2*log(2+exp(x1/2))-x2^2+0.5*(5-x3-x4)/5'                             
h=sym('[x1+x2-15]')                                                 
g1=sym('[2*log(2+exp(x1/2))-4]')
g2=sym('[2*x2+x2^2-20]') 
g3=sym('[-x1]')
g4=sym('[-x2]')
n=2,x1=7,x2=0,x3=0,x4=0,x5=0,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))
hold on


n=2,x1=5,x2=2,x3=3,x4=2,x5=0.3,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))
hold on

n=2,x1=0,x2=5,x3=5,x4=5,x5=0.5,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))
hold on



n=2,x1=2,x2=3,x3=4,x4=5,x5=0.7,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))
hold on

n=2,x1=11,x2=7,x3=6,x4=7,x5=1,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))
hold on 

n=2,x1=9,x2=8,x3=8,x4=9,x5=0.1,r1=[0.6],r2=[0.3],r3=[0.3],r4=[0.3],r5=[0.3],l=0.05,s=0.01            %
%z(1,1)=x1
%z(2,1)=x2
for j=1:500
    z(1,j)=x1;
    z(2,j)=x2;
    z(3,j)=x3;
    z(4,j)=x4;
    z(5,j)=x5;
    for i=1:5
        k=diff(f,['x',num2str(i)]);
        m=[diff(h,['x',num2str(i)])];
        n1=[diff(g1,['x',num2str(i)])];
        n2=[diff(g2,['x',num2str(i)])];

        v(i)=subs(k);
        p=subs(m);
        q1=subs(n1);
        q2=subs(n2);

        ch=subs(h);
        t1=(r1+ch)*p;                                                %
        t2=r2*q1;
        t3=r3*q2;

        u(i)=v(i)+t1+t2+t3;
    end
    x=[x1,x2,x3,x4,x5];
    u=x-u*l;                                                         %
    x1=u(1);
    x2=u(2);
    x3=u(3);
    x4=u(4);
    x5=u(5);
    
    o1=[subs(h)]';
    o2=[subs(g1)]';
    o3=[subs(g2)]';
    o4=[subs(g3)]';
    o5=[subs(g4)]';
    r1=r1+o1*s;                                                      %
    r2=r2+o2*s;
    r3=r3+o3*s;
    r4=r4+o4*s;
    r5=r5+o5*s;
end
j=1:500
plot(j,z(2,:))


xlabel('Iteration')
ylabel('variable state')
title('Trajectory of state vector')
