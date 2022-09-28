# ELK Observability Stack

Enabling observability via presence of `filebeat.yml`

## ELK stack config
Project setup `gregbugaj/marie-ai-elk` have been forked from `deviantony/docker-elk`.

```shell
docker compose down --volumes --remove-orphans  && docker compose up
```

## Filebeat setup

Install filebeat dashboards
```sh
./filebeat -c ./marie-ai/config/elk/filebeat.yml setup -e --dashboards
```

Test configuration 
```shell
./filebeat test config -c /home/gbugaj/dev/marie-ai/config/elk/filebeat.yml
```

Test output configuration 
```shell
./filebeat test output -c /home/gbugaj/dev/marie-ai/config/elk/filebeat.yml
```

```shell
logstash: localhost:5044...
  connection...
    parse host... OK
    dns lookup... OK
    addresses: 127.0.0.1
    dial up... OK
  TLS... WARN secure connection disabled
  talk to server... OK

```

## Alerting 
https://www.elastic.co/guide/en/kibana/8.4/alert-action-settings-kb.html#general-alert-action-settings


## References
https://github.com/gregbugaj/marie-ai-elk
https://www.elastic.co/guide/en/beats/filebeat/8.4/configuring-ssl-logstash.html#configuring-ssl-logstash
https://www.elastic.co/guide/en/beats/filebeat/current/logstash-output.html
https://www.elastic.co/guide/en/kibana/8.4/alert-action-settings-kb.html#general-alert-action-settings


https://kifarunix.com/easy-way-to-configure-filebeat-logstash-ssl-tls-connection/#test-logstash-configuration