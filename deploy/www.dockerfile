FROM nginx:1.17
WORKDIR /usr/share/nginx/html

COPY www/ .

EXPOSE 80
