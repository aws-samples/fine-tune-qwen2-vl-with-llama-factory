FROM vllm/vllm-openai:v0.9.0

COPY ./sagemaker-entrypoint.sh /app/
RUN chmod +x /app/sagemaker-entrypoint.sh

ENTRYPOINT ["/app/sagemaker-entrypoint.sh"]
