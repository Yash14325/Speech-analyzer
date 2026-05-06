FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Create non-root user and set ownership
RUN addgroup -S appuser && adduser -S appuser -G appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host"]
