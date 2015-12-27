import time

import pymysql
from tornado.options import options

from ramjet.settings import logger


if options.debug:
    DB_HOST = '10.32.189.146'  # czh
    DB_PASSWD = 'HFmWD7rDkfavyp2'
else:
    DB_HOST = '10.32.135.23'  # messerflow
    DB_PASSWD = 'CU3pbHkPzTudWU4'

PROJECTS_API = 'http://pm.dds.com/web/api/projects/jira/projects'
SCAN_INTERVAL_SEC = 12 * 3600

logger = logger.getChild('tasks.snmp-net')


class SyncProjectsError(Exception):
    pass


def bind_task(ioloop, thread_executor, process_executor):
    def run():
        logger.info('run')
        start = time.time()
        process_executor.submit(main)
        delay = max(SCAN_INTERVAL_SEC - (time.time() - start), 0)
        ioloop.call_later(delay, main)

    run()


def main():
    try:
        messerflow = connent2messerflow()
        jira = connent2jira()

        projects = fetch_projects(jira)
        flows = fetch_flows(jira)
        relations = fetch_relations(jira)
        if not projects or not flows or not relations:
            return

        relations_map = build_relations_map(relations, projects, flows)
        # update db
        clean_db(messerflow)
        update_flows(messerflow, flows)
        update_projects(messerflow, projects, relations_map)
    except Exception as err:
        logger.exception(err)


def connent2messerflow():
    return pymysql.connect(
        host=DB_HOST,
        password=DB_PASSWD,
        port=3306,
        db='messerflow',
        user='root',
        charset='utf8mb4'
    )


def connent2jira():
    return pymysql.connect(
        host='10.32.135.72',
        password='read_only',
        port=3306,
        db='jira',
        user='read_only',
        charset='utf8mb4'
    )


def fetch_db(connection, sql):
    with connection.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

    return result


def fetch_projects(connection):
    sql = 'SELECT id, pname, lead, description FROM project;'
    return fetch_db(connection, sql)


def fetch_flows(connection):
    sql = 'SELECT id, cname, description FROM projectcategory;'
    return fetch_db(connection, sql)


def fetch_relations(connection):
    sql = """
        SELECT source_node_id, sink_node_id
        FROM nodeassociation
        WHERE association_type='ProjectCategory';
    """
    return fetch_db(connection, sql)


def clean_db(messerflow):
    with messerflow.cursor() as cursor:
        cursor.execute('DELETE FROM messerflow.messerflow_ticket_projects')

    messerflow.commit()


def update_flows(connection, flows):
    sql = """
        INSERT INTO messerflow.messerflow_ticket_flows
        (flow_name, description)
        VALUES (%s, %s);
    """
    with connection.cursor() as cursor:
        args = [(cname, description) for id_, cname, description in flows]
        cursor.executemany(sql, args)

    connection.commit()


def build_relations_map(relations, projects, flows):
    pmap = {id_: pname for id_, pname, *_ in projects}
    fmap = {id_: fname for id_, fname, *_ in flows}
    return {pmap[pid]: fmap[fid] for pid, fid in relations}


def update_projects(connection, projects, relations_map):
    sql = """
        INSERT INTO messerflow.messerflow_ticket_projects
        (project_name, project_leader, description, project_flow)
        VALUES (%s, %s, %s, %s);
    """
    with connection.cursor() as cursor:
        args = [(pname, lead, description, relations_map.get(pname, '其他'))
                for id_, pname, lead, description in projects]
        cursor.executemany(sql, args)

    connection.commit()
